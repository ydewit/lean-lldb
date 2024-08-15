from __future__ import print_function, division
import sys
import logging
import lldb
import weakref

LOG = logging.getLogger(__name__)
MODULE = sys.modules[__name__]
CATEGORY = None

MAX_STRING_SUMMARY_LENGTH = 512


def initialize_category(debugger, internal_dict):
    global MODULE
    global CATEGORY

    CATEGORY = debugger.CreateCategory('Lean')
    CATEGORY.SetEnabled(True)

    # attach_synthetic_to_type(LeanSynthProvider, 'lean_object')
    attach_synthetic_to_type(LeanSynthProvider, 'lean_object *')

    # if 'lean' in internal_dict.get('source_languages', []):
    #     lldb.SBDebugger.SetInternalVariable('target.process.thread.step-avoid-regexp',
    #                                         '^<?(std|core|alloc)::', debugger.GetInstanceName())

    MAX_STRING_SUMMARY_LENGTH = debugger.GetSetting('target.max-string-summary-length').GetIntegerValue()


def attach_synthetic_to_type(synth_class, type_name, is_regex=False):
    global MODULE, CATEGORY
    # log.debug('attaching synthetic %s to "%s", is_regex=%s', synth_class.__name__, type_name, is_regex)
    synth = lldb.SBTypeSynthetic.CreateWithClassName(__name__ + '.' + synth_class.__name__)
    # synth.SetOptions(lldb.eTypeOptionCascade)
    synth.SetOptions(lldb.eTypeOptionNone)

    CATEGORY.AddTypeSynthetic(lldb.SBTypeNameSpecifier(type_name, is_regex), synth)

    def summary_fn(valobj, dict):
        return get_synth_summary(synth_class, valobj, dict)

    # LLDB accesses summary fn's by name, so we need to create a unique one.
    summary_fn.__name__ = '_get_synth_summary_' + synth_class.__name__
    setattr(MODULE, summary_fn.__name__, summary_fn)
    attach_summary_to_type(summary_fn, type_name, is_regex)


def attach_summary_to_type(summary_fn, type_name, is_regex=False):
    global MODULE, CATEGORY
    # log.debug('attaching summary %s to "%s", is_regex=%s', summary_fn.__name__, type_name, is_regex)
    summary = lldb.SBTypeSummary.CreateWithFunctionName(__name__ + '.' + summary_fn.__name__)
    # summary.SetOptions(lldb.eTypeOptionCascade)
    CATEGORY.AddTypeSummary(lldb.SBTypeNameSpecifier(type_name, is_regex), summary)


# 'get_summary' is annoyingly not a part of the standard LLDB synth provider API.
# This trick allows us to share data extraction logic between synth providers and their sibling summary providers.
def get_synth_summary(synth_class, valobj, dict):
    try:
        obj_id = valobj.GetIndexOfChildWithName('$$object-id$$')
        summary = LeanSynthProvider._get_synth_by_id(obj_id).get_summary()
        return str(summary)
    except KeyError as e:
        pass
    except Exception as e:
        LOG.exception('%s', e)
        raise


# Chained GetChildMemberWithName lookups
def gcm(valobj, *chain):
    for name in chain:
        valobj = valobj.GetChildMemberWithName(name)
    return valobj


# Get a pointer out of core::ptr::Unique<T>
def read_unique_ptr(valobj):
    pointer = valobj.GetChildMemberWithName('pointer')
    if pointer.TypeIsPointerType():  # Between 1.33 and 1.63 pointer was just *const T
        return pointer
    return pointer.GetChildAtIndex(0)


def string_from_ptr(pointer, length):
    if length <= 0:
        return u''
    error = lldb.SBError()
    process = pointer.GetProcess()
    data = process.ReadMemory(pointer.GetValueAsUnsigned(), length, error)
    if error.Success():
        return data.decode('utf8', 'replace')
    else:
        raise Exception('ReadMemory error: %s', error.GetCString())


def get_template_params(type_name):
    params = []
    level = 0
    start = 0
    for i, c in enumerate(type_name):
        if c == '<':
            level += 1
            if level == 1:
                start = i + 1
        elif c == '>':
            level -= 1
            if level == 0:
                params.append(type_name[start:i].strip())
        elif c == ',' and level == 1:
            params.append(type_name[start:i].strip())
            start = i + 1
    return params


def obj_summary(valobj, unavailable='{...}'):
    summary = valobj.GetSummary()
    if summary is not None:
        return summary
    summary = valobj.GetValue()
    if summary is not None:
        return summary
    return unavailable


def sequence_summary(childern, maxsize=32):
    s = ''
    for child in childern:
        if len(s) > 0:
            s += ', '
        s += obj_summary(child)
        if len(s) > maxsize:
            s += ', ...'
            break
    return s


def tuple_summary(obj, skip_first=0):
    fields = [obj_summary(obj.GetChildAtIndex(i)) for i in range(skip_first, obj.GetNumChildren())]
    return '(%s)' % ', '.join(fields)


# ----- Summaries -----

def tuple_summary_provider(valobj, dict={}):
    return tuple_summary(valobj)


# ----- Lean Constants ------

LEAN_MAX_CTOR_TAG = 244
LEAN_CLOSURE = 245
LEAN_ARRAY = 246
LEAN_STRUCT_ARRAY = 247
LEAN_SCALAR_ARRAY = 248
LEAN_STRING = 249
LEAN_MPZ = 250
LEAN_THUNK = 251
LEAN_TASK = 252
LEAN_REF = 253
LEAN_EXTERNAL = 254
LEAN_RESERVED = 255

LEAN_MAX_CTOR_FIELDS = 256
LEAN_MAX_CTOR_SCALARS_SIZE = 1024


# ----- Synth providers ------

# def init_static_vars ( cls, init, *vars ):
#     for var in vars:
#         if cls[var] is None:
#             cls[var] = init(var)

# typedef struct {
#     int      m_rc;
#     unsigned m_cs_sz:16;
#     unsigned m_other:8;
#     unsigned m_tag:8;
# } lean_object;
class LeanObjectSynthProvider(object):
    VOID_PTR_TYPE = None
    UINT8_T_TYPE = None
    UINT16_T_TYPE = None
    SIZE_T_TYPE = None
    CHAR_TYPE = None
    UNSIGNED_TYPE = None
    INT_TYPE = None
    LEAN_OBJECT_TYPE = None
    
    def _get_tag(valobj):
        # LOG.error('XXXXXX valobj: %s', valobj)
        return valobj.GetChildMemberWithName('m_tag').GetValueAsUnsigned()

    def _is_scalar(valobj):
        return valobj.GetValueAsUnsigned(0) & 1 == 1
    
    def _get_total_size(valobj):
        expr = '(size_t)lean_object_byte_size(%s)' % valobj.GetValueAsAddress()
        size = valobj.GetTarget().EvaluateExpression(expr)
        return size.GetValueAsUnsigned(0)

    def __init__(self, valobj, dict={}):
        # assert valobj.GetType().IsPointerType(), 'expected pointer type for %s' % valobj
        self.valobj = valobj
        self.m_rc = None
        self.m_cs_sz = None
        self.m_other = None
        self.m_tag = None

        # Initialize common static types (target may not be available otherwise)
        if LeanObjectSynthProvider.VOID_PTR_TYPE is None:
            LeanObjectSynthProvider.VOID_PTR_TYPE = self.valobj.GetTarget().FindFirstType('void').GetPointerType()
        
        if LeanObjectSynthProvider.UINT8_T_TYPE is None:
            LeanObjectSynthProvider.UINT8_T_TYPE = self.valobj.GetTarget().FindFirstType('uint8_t')

        if LeanObjectSynthProvider.UINT16_T_TYPE is None:
            LeanObjectSynthProvider.UINT16_T_TYPE = self.valobj.GetTarget().FindFirstType('uint16_t')
        
        if LeanObjectSynthProvider.SIZE_T_TYPE is None:
            LeanObjectSynthProvider.SIZE_T_TYPE = self.valobj.GetTarget().FindFirstType('size_t')

        if LeanObjectSynthProvider.CHAR_TYPE is None:
            LeanObjectSynthProvider.CHAR_TYPE = self.valobj.GetTarget().FindFirstType('char')
        
        if LeanObjectSynthProvider.UNSIGNED_TYPE is None:
            LeanObjectSynthProvider.UNSIGNED_TYPE = self.valobj.GetTarget().FindFirstType('unsigned')

        if LeanObjectSynthProvider.INT_TYPE is None:
            LeanObjectSynthProvider.INT_TYPE = self.valobj.GetTarget().FindFirstType('int')

        if LeanObjectSynthProvider.LEAN_OBJECT_TYPE is None:
            LeanObjectSynthProvider.LEAN_OBJECT_TYPE = self.valobj.GetTarget().FindFirstType('lean_object')

        self.update()

    # ----- lean_object fields ------

    def get_tag(self): # unsigned
        return self.m_tag.GetValueAsUnsigned(0)
    
    def get_other(self): # unsigned
        return self.m_other.GetValueAsUnsigned(0)

    def get_rc(self): # signed
        return self.m_rc.GetValueAsSigned(0)
    
    def get_cs_sz(self): # unsigned
        return self.m_cs_sz.GetValueAsUnsigned(0)

    ## helpers
    def is_scalar(self):
        return LeanObjectSynthProvider._is_scalar(self.valobj)

    def get_void_ptr_type(self):
        assert LeanObjectSynthProvider.VOID_PTR_TYPE is not None, 'void* type not initialized'
        return LeanObjectSynthProvider.VOID_PTR_TYPE
        
    def get_uint8_t_type(self):
        assert LeanObjectSynthProvider.UINT8_T_TYPE is not None, 'uint8_t type not initialized'
        return LeanObjectSynthProvider.UINT8_T_TYPE

    def get_uint16_t_type(self):
        assert LeanObjectSynthProvider.UINT16_T_TYPE is not None, 'uint16_t type not initialized'
        return LeanObjectSynthProvider.UINT16_T_TYPE
    
    def get_size_t_type(self):
        assert LeanObjectSynthProvider.SIZE_T_TYPE is not None, 'size_t type not initialized'
        return LeanObjectSynthProvider.SIZE_T_TYPE
    
    def get_char_type(self):
        assert LeanObjectSynthProvider.CHAR_TYPE is not None, 'char type bytes not initialized'
        return LeanObjectSynthProvider.CHAR_TYPE
    
    def get_unsigned_type(self):
        assert LeanObjectSynthProvider.UNSIGNED_TYPE is not None, 'unsigned type not initialized'
        return LeanObjectSynthProvider.UNSIGNED_TYPE
    
    def get_int_type(self):
        assert LeanObjectSynthProvider.INT_TYPE is not None, 'int type not initialized'
        return LeanObjectSynthProvider.INT_TYPE
    
    def get_lean_object_type(self):
        assert LeanObjectSynthProvider.LEAN_OBJECT_TYPE is not None, 'lean_object type not initialized'
        return LeanObjectSynthProvider.LEAN_OBJECT_TYPE
    
    def get_body_address_of(self):
        return self.valobj.GetValueAsAddress() + self.get_lean_object_type().GetByteSize()
    
    def cast(self, type_name):
        type = self.valobj.GetTarget().FindFirstType(type_name).GetPointerType()
        return self.valobj.Cast(type)

    def get_addr_size(self): # uint8_t
        return self.valobj.GetTarget().GetAddressByteSize()

    def get_type(self): # SBType
        return self.valobj.GetType()

    def _call(self, name, returns, *args):
        argStrs = [ ]
        for arg in args:
            if arg is None:
                raise Exception('None argument')
            elif arg.IsValid() == False:
                raise Exception('Invalid argument')
            elif isinstance(arg, str):
                LOG.warn('str argument type: %s', arg.GetValue())
                argStrs.append('"%s"' % arg.replace('\\', '\\\\').replace('"', '\"'))
            else:
                argStrs.append(arg.GetValue())
        expr = '(%s)%s(%s)' % (returns, name, ', '.join(argStrs))
        LOG.error('XXXXXX expr: %s', expr)
        return self.valobj.GetFrame().EvaluateExpression(expr)

    # ----- Synth interface ------

    # update the internal state whenever the state of the variables in LLDB changes
    def update(self):
        self.m_rc = None
        self.m_cs_sz = None
        self.m_other = None
        self.m_tag = None
        try:
            self.m_rc    = self.valobj.GetChildMemberWithName('m_rc')
            # self.m_rc    = self.valobj.GetValueForExpressionPath('->m_rc')
            self.m_cs_sz = self.valobj.GetChildMemberWithName('m_cs_sz')
            # self.m_cs_sz = self.valobj.GetValueForExpressionPath('->m_cs_sz')
            self.m_other = self.valobj.GetChildMemberWithName('m_other').Clone('m_other')
            # self.m_other = self.valobj.GetValueForExpressionPath('->m_other')
            self.m_tag   = self.valobj.GetChildMemberWithName('m_tag')
            # self.m_tag   = self.valobj.GetValueForExpressionPath('->m_tag')
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    # True if this object might have children, and False if this object can be guaranteed not to have children.
    def has_children(self):
        return True
        # return False

    # the number of children that you want your object to have 
    def num_children(self):
        return 4

    # return a new LLDB SBValue object representing the child at the index given as argument 
    def get_child_at_index(self, index): # return SBValue for the child
        if index == 0:
            return self.m_rc
        elif index == 1:
            return self.m_cs_sz
        elif index == 2:
            return self.m_other
        elif index == 3:
            return self.m_tag
        return None

    # the index of the synthetic child whose name is given as argument 
    def get_child_index(self, name):
        if name == 'm_rc':
            return 0
        elif name == 'm_cs_sz':
            return 1
        elif name == 'm_other':
            return 2
        elif name == 'm_tag':
            return 3
        else:
            return -1            
        # LOG.error('XXXXXX name: %s', name)
        # if name.startswith('['):
        #     return int(name.lstrip('[').rstrip(']'))
        # return -1

    def get_summary(self):
        return "{Obj|TAG=%d,RC=%s,OTHER=%u,CS_SZ=%s}" % (self.get_rc(), self.get_tag() if self.get_rc() != 0 else "∞", self.get_other(), self.get_cs_sz())


class LeanBoxedScalarSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj, dict)
        self.update()

    def _box_scalar(valobj):
        return valobj.CreateValueFromExpression(None, '((lean_object*)(%s << 1 | 1))' % valobj.GetValueAsUnsigned())
    
    def _unbox_scalar(valobj):
        return valobj.GetValueAsUnsigned() >> 1
    
    def box_scalar(self):
        return LeanBoxedScalarSynthProvider._box_scalar(self.valobj)

    def unbox_scalar(self):
        return LeanBoxedScalarSynthProvider._unbox_scalar(self.valobj)

    # ----- Synth interface ------

    def has_children(self):
        return False

    # the number of children that you want your object to have 
    def num_children(self):
        return 0

    def get_summary(self):
        try:
            return "{Boxed} %d" % self.unbox_scalar()
        except Exception as e:
            LOG.exception('%s', e)
            return "error"

# typedef struct {
#     lean_object   m_header;
#     lean_object * m_objs[0];
# } lean_ctor_object;
class LeanCtorSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag <= LEAN_MAX_CTOR_TAG, 'invalid ctor object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_ctor_idx(self):
        return self.get_tag()

    def get_num_objs(self):
        return self.get_other()

    def get_objs(self):
        return self.m_objs

    # scalar fields: AFAIK, there is not enough runtime info to retrieve them!
    def has_scalars(self):
        objs_size = self.get_num_objs() * self.get_lean_object_type().GetPointerType().GetByteSize()
        return LeanObjectSynthProvider._get_total_size(self.valobj) - self.get_lean_object_type().GetByteSize() - objs_size > 0

    # ----- SynthProvider interface ------

    def update(self):
        try:
            super().update()
            self.m_objs = self.valobj.CreateValueFromAddress("m_objs", self.get_body_address_of(), self.get_lean_object_type().GetPointerType().GetPointerType())
            assert self.m_objs.IsValid()
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def has_children(self):
        return True
    
    def num_children(self):
        if self.get_num_objs() > LEAN_MAX_CTOR_FIELDS:
            return LEAN_MAX_CTOR_FIELDS
        return self.get_num_objs()
    
    def get_child_at_index(self, index):
        if index >= self.get_num_objs():
            return None
        if index < 0:
            return None

        type = self.get_lean_object_type().GetPointerType()
        offset = index * type.GetByteSize()
        return self.get_objs().CreateChildAtOffset('[%s]' % index, offset, type)

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return -1
    
    def get_summary(self):
        return "{Ctor#%u|RC=%s} objs=%u, scalars=%s" % (self.get_ctor_idx(), str(self.get_rc()) if self.get_rc() != 0 else "∞", self.get_num_objs(), "true" if self.has_scalars() else "false")

# typedef struct {
#     lean_object   m_header;
#     void *        m_fun;
#     uint16_t      m_arity;     /* Number of arguments expected by m_fun. */
#     uint16_t      m_num_fixed; /* Number of arguments that have been already fixed. */
#     lean_object * m_objs[0];
# } lean_closure_object;
class LeanClosureSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        super().__init__(valobj, dict)
        self.update()
        assert tag == LEAN_CLOSURE, 'invalid closure object %d' % tag

    # ----- fields ------

    def get_fun(self): # SBFunction
        return self.m_fun
    
    def get_arity(self): # unsigned
        return self.m_arity.GetValueAsUnsigned()
    
    def get_num_fixed(self): # unsigned
        return self.m_num_fixed.GetValueAsUnsigned()
    
    def get_objs(self):
        return self.m_objs
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_fun = None
        self.m_arity = None
        self.m_num_fixed = None
        self.m_objs = None
        try:
            super().update()
            self.m_fun       = self.valobj.CreateValueFromAddress("m_fun",       self.get_body_address_of(), self.get_void_ptr_type())
            self.m_arity     = self.valobj.CreateValueFromAddress("m_arity",     self.get_body_address_of() + self.get_void_ptr_type().GetByteSize(), self.get_uint16_t_type())
            self.m_num_fixed = self.valobj.CreateValueFromAddress("m_num_fixed", self.get_body_address_of() + self.get_void_ptr_type().GetByteSize() + self.get_uint16_t_type().GetByteSize(), self.get_uint16_t_type())
            self.m_objs      = self.valobj.CreateValueFromAddress("m_objs",      self.get_body_address_of() + self.get_void_ptr_type().GetByteSize() + self.get_uint16_t_type().GetByteSize() + self.get_uint16_t_type().GetByteSize(), self.get_lean_object_type().GetPointerType())
        except Exception as e:
            LOG.error('update failed: %s', e)
            pass

    def has_children(self):
        return self.get_num_fixed() > 0

    def num_children(self):
        return self.get_num_fixed()
    
    def get_child_at_index(self, index):
        if index >= self.get_num_fixed():
            return None
        if index < 0:
            return None
        type = self.get_lean_object_type().GetPointerType()
        offset = index * type.GetByteSize()
        return self.get_objs().CreateChildAtOffset('[%s]' % index, offset, type)
    
    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return -1

    def get_summary(self):
        return "{Closure|RC=%d} fun=%s, arity=%s, num_fixed=%u" % (self.get_rc(), hex(self.get_fun().GetLoadAddress()), self.get_arity(), self.get_num_fixed())


# /* Array arrays */
# typedef struct {
#     lean_object   m_header;
#     size_t        m_size;
#     size_t        m_capacity;
#     lean_object * m_data[0];
# } lean_array_object;
class LeanArraySynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_ARRAY, 'invalid array object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_size(self): # unsigned
        return self.m_size.GetValueAsUnsigned()
    
    def get_capacity(self): # unsigned
        return self.m_capacity.GetValueAsUnsigned()
    
    def get_data(self): # SBValue
        return self.m_data
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_size = None
        self.m_capacity = None
        self.m_data = None
        try:
            super().update()
            self.m_size     = self.valobj.CreateValueFromAddress("m_size",     self.get_body_address_of(), self.get_size_t_type())
            self.m_capacity = self.valobj.CreateValueFromAddress("m_capacity", self.get_body_address_of() + 1 * self.get_size_t_type().GetByteSize(), self.get_size_t_type())
            self.m_data     = self.valobj.CreateValueFromAddress("m_data",     self.get_body_address_of() + 2 * self.get_size_t_type().GetByteSize(), self.get_lean_object_type().GetPointerType())
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def has_children(self):
        return self.get_size() > 0
    
    def num_children(self):
        return self.get_size()
    
    def get_child_at_index(self, index): # FIXME!
        if not 0 <= index < self.get_size():
            return None
        type = self.get_lean_object_type().GetPointerType()
        offset = type.GetByteSize() * index
        return self.get_data().CreateChildAtOffset('[%s]' % index, offset, type)

    def get_child_index(self, name):
        try:
            return int(name.lstrip("[").rstrip("]"))
        except:
            return -1

    def get_summary(self):
        return "{Array|RC=%d} size=%u, capacity=%u" % (self.get_rc(), self.get_size(), self.get_capacity())

class LeanStructArraySynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):        
        assert tag == LEAN_STRUCT_ARRAY, 'invalid struct array object %d' % tag
        super().__init__(valobj, dict)

    def has_children(self):
        return 0
    
    def get_summary(self):
        return "<invalid struct array object>"
    

# /* Scalar arrays */
# typedef struct {
#     lean_object   m_header;
#     size_t        m_size;
#     size_t        m_capacity;
#     uint8_t       m_data[0];
# } lean_sarray_object;
class LeanScalarArraySynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_SCALAR_ARRAY, 'invalid scalar array object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_size(self):
        return self.m_size.GetValueAsUnsigned()
    
    def get_capacity(self):
        return self.m_capacity.GetValueAsUnsigned()
    
    def get_data(self):
        return self.m_data

    # ----- SynthProvider interface ------

    def update(self):
        self.m_size = None
        self.m_capacity = None
        self.m_data = None
        try:
            super().update()
            self.m_size     = self.valobj.CreateValueFromAddress("m_size",     self.get_body_address_of(), self.get_size_t_type())
            self.m_capacity = self.valobj.CreateValueFromAddress("m_capacity", self.get_body_address_of() + 1 * self.get_size_t_type().GetByteSize(), self.get_size_t_type())
            self.m_data     = self.valobj.CreateValueFromAddress("m_data",     self.get_body_address_of() + 2 * self.get_size_t_type().GetByteSize(), self.get_uint8_t_type())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def has_children(self):
        return self.get_size() > 0
    
    def num_children(self):
        return self.get_size()
    
    def get_child_at_index(self, index):
        return LeanBoxedScalarSynthProvider._box_scalar(self.get_data().GetChildAtIndex(index).GetValueAsUnsigned(0))
    
    def get_summary(self):
        return "{ScalarArray|RC=%d} size=%u, capacity=%u, [%s])" % (self.get_rc(), self.get_size())


# typedef struct {
#     lean_object m_header;
#     size_t      m_size;     /* byte length including '\0' terminator */
#     size_t      m_capacity;
#     size_t      m_length;   /* UTF8 length */
#     char        m_data[0];
# } lean_string_object;
class LeanStringSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_STRING, 'invalid string object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_size(self):
        return self.m_size.GetValueAsUnsigned()
    
    def get_capacity(self):
        return self.m_capacity.GetValueAsUnsigned()
    
    def get_length(self):
        return self.m_length.GetValueAsUnsigned()
    
    def get_data(self):
        return self.m_data

    def get_data_as_string(self):
        addr = self.get_data().AddressOf().GetValueAsUnsigned()
        # if addr == 0:
        #     return u'NULL'
        error = lldb.SBError()
        content = self.valobj.process.ReadCStringFromMemory(addr, MAX_STRING_SUMMARY_LENGTH, error)
        if error.Success():
            if self.get_length() > MAX_STRING_SUMMARY_LENGTH:
                return '"%s..."' % content
            else:
                return '"%s"' % content
        else:
            return "<error: %s>" % error.GetCString()

    # ----- SynthProvider interface ------

    def update(self):
        self.m_size = None
        self.m_capacity = None
        self.m_length = None
        self.m_data = None
        try:
            super().update()
            self.m_size     = self.valobj.CreateValueFromAddress("m_size",     self.get_body_address_of(), self.get_size_t_type())
            self.m_capacity = self.valobj.CreateValueFromAddress("m_capacity", self.get_body_address_of() + 1 * self.get_size_t_type().GetByteSize(), self.get_size_t_type())
            self.m_length   = self.valobj.CreateValueFromAddress("m_length",   self.get_body_address_of() + 2 * self.get_size_t_type().GetByteSize(), self.get_size_t_type())
            self.m_data     = self.valobj.CreateValueFromAddress("m_data",     self.get_body_address_of() + 3 * self.get_size_t_type().GetByteSize(), self.get_char_type().GetPointerType())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def get_summary(self):
        return "{String|RC=%d} size=%u (%u), capacity=%u, %s" % (self.get_rc(), self.get_size(), self.get_length(), self.get_capacity(), self.get_data_as_string())

class LeanMpzSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_MPZ, 'invalid mpz object %d' % tag
        super().__init__(valobj, dict)

    def has_children(self):
        return 0
    
    def get_summary(self):
        return "<invalid mpz object>"

# typedef struct {
#     lean_object            m_header;
#     _Atomic(lean_object *) m_value;
#     _Atomic(lean_object *) m_closure;
# } lean_thunk_object;
class LeanThunkSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_THUNK, 'invalid thunk object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_value(self):
        return self.m_value
    
    def get_closure(self):
        return self.m_closure
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_value = None
        self.m_closure = None
        try:
            super().update()
            self.m_value     = self.valobj.CreateValueFromAddress("m_value",   self.get_body_address_of(), self.get_lean_object_type().GetPointerType())
            self.m_closure = self.valobj.CreateValueFromAddress("m_closure", self.get_body_address_of() + self.get_lean_object_type().GetByteSize(), self.get_lean_object_type().GetPointerType())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def get_summary(self):
        if self.get_value().GetValueAsAddress() == 0:
            return "{Thunk|rc=%s} %s" % (self.get_rc(), 
            self.get_closure().GetSummary())
        else:
            return "{Thunk|RC=%s} %s" % (self.get_rc(), self.get_value().GetSummary())



# typedef struct lean_task {
#     lean_object            m_header;
#     _Atomic(lean_object *) m_value;
#     lean_task_imp *        m_imp;
# } lean_task_object;
class LeanTaskSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_TASK, 'invalid task object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- SynthProvider interface ------

    def get_value(self):
        return self.m_value
    
    def get_imp(self):
        return self.m_imp
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_value = None
        self.m_imp = None
        try:
            super().update()
            self.m_value = self.valobj.CreateValueFromAddress("m_value", self.get_body_address_of(), self.get_lean_object_type().GetPointerType())
            self.m_imp   = self.valobj.CreateValueFromAddress("m_imp",   self.get_body_address_of() + self.get_lean_object_type().GetByteSize(), self.get_void_ptr_type())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def get_summary(self):
        return "{Task|RC=%d} value=%s, impl=%s)" % (self.get_rc(), self.get_value().GetSummary(), hex(self.get_imp().GetLoadAddress()))

# typedef struct {
#     lean_object   m_header;
#     lean_object * m_value;
# } lean_ref_object;
class LeanRefSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_REF, 'invalid ref object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # fields

    def get_value(self):
        return self.m_value
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_value = None
        try:
            super().update()
            self.m_value = self.valobj.CreateValueFromAddress("m_value", self.get_body_address_of(), self.get_lean_object_type().GetPointerType())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def get_summary(self):
        return "(Ref|RC=%d} %s" % (self.get_rc(), self.get_value().GetSummary())


# typedef struct {
#     lean_external_finalize_proc m_finalize;
#     lean_external_foreach_proc  m_foreach;
# } lean_external_class;
#
# /* Object for wrapping external data. */
# typedef struct {
#     lean_object           m_header;
#     lean_external_class * m_class;
#     void *                m_data;
# } lean_external_object;
class LeanExternalSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):
        assert tag == LEAN_EXTERNAL, 'invalid external object %d' % tag
        super().__init__(valobj, dict)
        self.update()

    # ----- fields ------

    def get_class(self):
        return self.m_class
    
    def get_data(self):
        return self.m_data
    
    # ----- SynthProvider interface ------

    def update(self):
        self.m_class = None
        self.m_data = None
        try:
            super().update()
            self.m_class = self.valobj.CreateValueFromAddress("m_class", self.get_body_address_of(), self.get_void_ptr_type())
            self.m_data  = self.valobj.CreateValueFromAddress("m_data",  self.get_body_address_of() + self.get_void_ptr_type().GetByteSize(), self.get_void_ptr_type())
            return True
        except Exception as e:
            LOG.warning('update failed: %s', e)
            pass

    def get_summary(self):
        return "{External|RC=%d} class=%s, data=%s" % (self.get_rc(), hex(self.get_value().GetLoadAddress()), hex(self.get_data().GetLoadAddress()))


class LeanReservedSynthProvider(LeanObjectSynthProvider):
    def __init__ (self, tag, valobj, dict={}):        
        assert tag == LEAN_RESERVED, 'invalid reserved object %d' % tag
        super().__init__(valobj, dict)

    def has_children(self):
        return 0
    
    def get_summary(self):
        return "<invalid object>"

class LeanSynthProvider(object):
    _synth_by_id = weakref.WeakValueDictionary()
    _next_id = 0

    def _get_next_id():
        obj_id = LeanSynthProvider._next_id
        LeanSynthProvider._next_id += 1
        return obj_id
    
    def _get_synth_by_id(id):
        provider = LeanSynthProvider._synth_by_id[id]
        return provider
    
    def _add_synth_by_id(provider):
        # LOG.error('provider.obj_id: %d %s', provider.m_obj_id, str(type(provider.m_obj)))
        LeanSynthProvider._synth_by_id[provider.m_obj_id] = provider
        return provider.m_obj_id

    def __init__(self, valobj, dict={}):
        if LeanObjectSynthProvider._is_scalar(valobj):
            self.m_tag = -1
            self.m_obj = LeanBoxedScalarSynthProvider(valobj, dict)
        else:
            self.m_tag = LeanObjectSynthProvider._get_tag(valobj)

            if self.m_tag <= LEAN_MAX_CTOR_TAG:
                self.m_obj = LeanCtorSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_CLOSURE:
                self.m_obj = LeanClosureSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_ARRAY:
                self.m_obj = LeanArraySynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_STRUCT_ARRAY:
                self.m_obj = LeanStructArraySynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_SCALAR_ARRAY:
                self.m_obj = LeanScalarArraySynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_STRING:
                self.m_obj = LeanStringSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_MPZ:
                self.m_obj = LeanMpzSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_THUNK:
                self.m_obj = LeanThunkSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_TASK:
                self.m_obj = LeanTaskSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_REF:
                self.m_obj = LeanRefSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_EXTERNAL:
                self.m_obj = LeanExternalSynthProvider(self.m_tag, valobj, dict)
            elif self.m_tag == LEAN_RESERVED:
                self.m_obj = LeanReservedSynthProvider(self.m_tag, valobj, dict)
            else:
                raise Exception('Unknown lean object tag: %d for %s' % (self.m_tag, valobj))

        self.m_obj_id = LeanSynthProvider._get_next_id()
        LeanSynthProvider._add_synth_by_id(self)

    
    def update(self):
        # LOG.error('updated called!')
        return self.m_obj.update()

    def has_children(self):
        return self.m_obj.has_children()

    def num_children(self):
        return self.m_obj.num_children()

    def get_child_at_index(self, index):
        return self.m_obj.get_child_at_index(index)

    def get_child_index(self, name):
        if name == '$$object-id$$':
            return self.m_obj_id
        return self.m_obj.get_child_index(name)

    def get_summary(self):
        return self.m_obj.get_summary()

##################################################################################################################

class RustSynthProvider(object):
    synth_by_id = weakref.WeakValueDictionary()
    next_id = 0

    def __init__(self, valobj, dict={}):
        self.valobj = valobj
        self.obj_id = RustSynthProvider.next_id
        RustSynthProvider.synth_by_id[self.obj_id] = self
        RustSynthProvider.next_id += 1

    def update(self):
        return True

    def has_children(self):
        return False

    def num_children(self):
        return 0

    def get_child_at_index(self, index):
        return None

    def get_child_index(self, name):
        if name == '$$object-id$$':
            return self.obj_id

        try:
            return self.get_index_of_child(name)
        except Exception as e:
            LOG.exception('%s', e)
            raise

    def get_summary(self):
        return None


class ArrayLikeSynthProvider(RustSynthProvider):
    '''Base class for providers that represent array-like objects'''

    def update(self):
        self.ptr, self.len = self.ptr_and_len(self.valobj)  # type: ignore
        self.item_type = self.ptr.GetType().GetPointeeType()
        self.item_size = self.item_type.GetByteSize()

    def ptr_and_len(self, obj):
        pass  # abstract

    def num_children(self):
        return self.len

    def has_children(self):
        return True

    def get_child_at_index(self, index):
        try:
            if not 0 <= index < self.len:
                return None
            offset = index * self.item_size
            return self.ptr.CreateChildAtOffset('[%s]' % index, offset, self.item_type)
        except Exception as e:
            LOG.exception('%s', e)
            raise

    def get_index_of_child(self, name):
        return int(name.lstrip('[').rstrip(']'))

    def get_summary(self):
        return '(%d)' % (self.len,)


class StdVectorSynthProvider(ArrayLikeSynthProvider):
    def ptr_and_len(self, vec):
        return (
            read_unique_ptr(gcm(vec, 'buf', 'ptr')),
            gcm(vec, 'len').GetValueAsUnsigned()
        )

    def get_summary(self):
        return '(%d) vec![%s]' % (self.len, sequence_summary((self.get_child_at_index(i) for i in range(self.len))))


class StdVecDequeSynthProvider(RustSynthProvider):
    def update(self):
        self.ptr = read_unique_ptr(gcm(self.valobj, 'buf', 'ptr'))
        self.cap = gcm(self.valobj, 'buf', 'cap').GetValueAsUnsigned()

        head = gcm(self.valobj, 'head').GetValueAsUnsigned()

        # rust 1.67 changed from a head, tail implementation to a head, length impl
        # https://github.com/rust-lang/rust/pull/102991
        vd_len = gcm(self.valobj, 'len')
        if vd_len.IsValid():
            self.len = vd_len.GetValueAsUnsigned()
            self.startptr = head
        else:
            tail = gcm(self.valobj, 'tail').GetValueAsUnsigned()
            self.len = head - tail
            self.startptr = tail

        self.item_type = self.ptr.GetType().GetPointeeType()
        self.item_size = self.item_type.GetByteSize()

    def num_children(self):
        return self.len

    def has_children(self):
        return True

    def get_child_at_index(self, index):
        try:
            if not 0 <= index < self.num_children():
                return None
            offset = ((self.startptr + index) % self.cap) * self.item_size
            return self.ptr.CreateChildAtOffset('[%s]' % index, offset, self.item_type)
        except Exception as e:
            LOG.exception('%s', e)
            raise

    def get_index_of_child(self, name):
        return int(name.lstrip('[').rstrip(']'))

    def get_summary(self):
        return '(%d) VecDeque[%s]' % (self.num_children(), sequence_summary((self.get_child_at_index(i) for i in range(self.num_children()))))

##################################################################################################################


class SliceSynthProvider(ArrayLikeSynthProvider):
    def ptr_and_len(self, vec):
        return (
            gcm(vec, 'data_ptr'),
            gcm(vec, 'length').GetValueAsUnsigned()
        )

    def get_summary(self):
        return '(%d) &[%s]' % (self.len, sequence_summary((self.get_child_at_index(i) for i in range(self.len))))


class MsvcSliceSynthProvider(SliceSynthProvider):
    def get_type_name(self):
        tparams = get_template_params(self.valobj.GetTypeName())
        return '&[' + tparams[0] + ']'


# Base class for *String providers
class StringLikeSynthProvider(ArrayLikeSynthProvider):
    def get_child_at_index(self, index):
        ch = ArrayLikeSynthProvider.get_child_at_index(self, index)
        ch.SetFormat(lldb.eFormatChar)
        return ch

    def get_summary(self):
        strval = string_from_ptr(self.ptr, min(self.len, MAX_STRING_SUMMARY_LENGTH))
        if self.len > MAX_STRING_SUMMARY_LENGTH:
            strval += u'...'
        return u'"%s"' % strval


class StrSliceSynthProvider(StringLikeSynthProvider):
    def ptr_and_len(self, valobj):
        return (
            gcm(valobj, 'data_ptr'),
            gcm(valobj, 'length').GetValueAsUnsigned()
        )

    def get_type_name(self):
        return '&str'

class StdStringSynthProvider(StringLikeSynthProvider):
    def ptr_and_len(self, valobj):
        vec = gcm(valobj, 'vec')
        return (
            read_unique_ptr(gcm(vec, 'buf', 'ptr')),
            gcm(vec, 'len').GetValueAsUnsigned()
        )


class StdCStringSynthProvider(StringLikeSynthProvider):
    def ptr_and_len(self, valobj):
        vec = gcm(valobj, 'inner')
        return (
            gcm(vec, 'data_ptr'),
            gcm(vec, 'length').GetValueAsUnsigned() - 1
        )


class StdOsStringSynthProvider(StringLikeSynthProvider):
    def ptr_and_len(self, valobj):
        vec = gcm(valobj, 'inner', 'inner')
        tmp = gcm(vec, 'bytes')  # Windows OSString has an extra layer
        if tmp.IsValid():
            vec = tmp
        return (
            read_unique_ptr(gcm(vec, 'buf', 'ptr')),
            gcm(vec, 'len').GetValueAsUnsigned()
        )


class FFISliceSynthProvider(StringLikeSynthProvider):
    def ptr_and_len(self, valobj):
        process = valobj.GetProcess()
        slice_ptr = valobj.GetLoadAddress()
        data_ptr_type = valobj.GetTarget().GetBasicType(lldb.eBasicTypeChar).GetPointerType()
        # Unsized slice objects have incomplete debug info, so here we just assume standard slice
        # reference layout: [<pointer to data>, <data size>]
        error = lldb.SBError()
        pointer = valobj.CreateValueFromAddress('data', slice_ptr, data_ptr_type)
        length = process.ReadPointerFromMemory(slice_ptr + process.GetAddressByteSize(), error)
        return pointer, length


class StdCStrSynthProvider(FFISliceSynthProvider):
    def ptr_and_len(self, valobj):
        ptr, len = FFISliceSynthProvider.ptr_and_len(self, valobj)
        return (ptr, len-1)  # drop terminaing '\0'


class StdOsStrSynthProvider(FFISliceSynthProvider):
    pass


class StdPathBufSynthProvider(StdOsStringSynthProvider):
    def ptr_and_len(self, valobj):
        return StdOsStringSynthProvider.ptr_and_len(self, gcm(valobj, 'inner'))


class StdPathSynthProvider(FFISliceSynthProvider):
    pass

##################################################################################################################


class DerefSynthProvider(RustSynthProvider):
    deref = lldb.SBValue()

    def has_children(self):
        return self.deref.MightHaveChildren()

    def num_children(self):
        return self.deref.GetNumChildren()

    def get_child_at_index(self, index):
        return self.deref.GetChildAtIndex(index)

    def get_index_of_child(self, name):
        return self.deref.GetIndexOfChildWithName(name)

    def get_summary(self):
        return obj_summary(self.deref)

# Base for Rc and Arc


class StdRefCountedSynthProvider(DerefSynthProvider):
    weak = 0
    strong = 0

    def get_summary(self):
        if self.weak != 0:
            s = '(refs:%d,weak:%d) ' % (self.strong, self.weak)
        else:
            s = '(refs:%d) ' % self.strong
        if self.strong > 0:
            s += obj_summary(self.deref)
        else:
            s += '<disposed>'
        return s


class StdRcSynthProvider(StdRefCountedSynthProvider):
    def update(self):
        inner = read_unique_ptr(gcm(self.valobj, 'ptr'))
        self.strong = gcm(inner, 'strong', 'value', 'value').GetValueAsUnsigned()
        self.weak = gcm(inner, 'weak', 'value', 'value').GetValueAsUnsigned()
        if self.strong > 0:
            self.deref = gcm(inner, 'value')
            self.weak -= 1  # There's an implicit weak reference communally owned by all the strong pointers
        else:
            self.deref = lldb.SBValue()
        self.deref.SetPreferSyntheticValue(True)


class StdArcSynthProvider(StdRefCountedSynthProvider):
    def update(self):
        inner = read_unique_ptr(gcm(self.valobj, 'ptr'))
        self.strong = gcm(inner, 'strong', 'v', 'value').GetValueAsUnsigned()
        self.weak = gcm(inner, 'weak', 'v', 'value').GetValueAsUnsigned()
        if self.strong > 0:
            self.deref = gcm(inner, 'data')
            self.weak -= 1  # There's an implicit weak reference communally owned by all the strong pointers
        else:
            self.deref = lldb.SBValue()
        self.deref.SetPreferSyntheticValue(True)


class StdMutexSynthProvider(DerefSynthProvider):
    def update(self):
        self.deref = gcm(self.valobj, 'data', 'value')
        self.deref.SetPreferSyntheticValue(True)


class StdCellSynthProvider(DerefSynthProvider):
    def update(self):
        self.deref = gcm(self.valobj, 'value', 'value')
        self.deref.SetPreferSyntheticValue(True)


class StdRefCellSynthProvider(DerefSynthProvider):
    def update(self):
        self.deref = gcm(self.valobj, 'value', 'value')
        self.deref.SetPreferSyntheticValue(True)

    def get_summary(self):
        borrow = gcm(self.valobj, 'borrow', 'value', 'value').GetValueAsSigned()
        s = ''
        if borrow < 0:
            s = '(borrowed:mut) '
        elif borrow > 0:
            s = '(borrowed:%d) ' % borrow
        return s + obj_summary(self.deref)


class StdRefCellBorrowSynthProvider(DerefSynthProvider):
    def update(self):
        self.deref = gcm(self.valobj, 'value', 'pointer').Dereference()
        self.deref.SetPreferSyntheticValue(True)

##################################################################################################################


class EnumSynthProvider(RustSynthProvider):
    variant = lldb.SBValue()
    summary = ''
    skip_first = 0

    def has_children(self):
        return self.variant.MightHaveChildren()

    def num_children(self):
        return self.variant.GetNumChildren() - self.skip_first

    def get_child_at_index(self, index):
        return self.variant.GetChildAtIndex(index + self.skip_first)

    def get_index_of_child(self, name):
        return self.variant.GetIndexOfChildWithName(name) - self.skip_first

    def get_summary(self):
        return self.summary


class GenericEnumSynthProvider(EnumSynthProvider):
    def update(self):
        dyn_type_name = self.valobj.GetTypeName()
        variant_name = dyn_type_name[dyn_type_name.rfind(':')+1:]
        self.variant = self.valobj

        if self.variant.IsValid() and self.variant.GetNumChildren() > self.skip_first:
            if self.variant.GetChildAtIndex(self.skip_first).GetName() in ['0', '__0']:
                self.summary = variant_name + tuple_summary(self.variant)
            else:
                self.summary = variant_name + '{...}'
        else:
            self.summary = variant_name


class MsvcTupleSynthProvider(RustSynthProvider):
    def update(self):
        tparams = get_template_params(self.valobj.GetTypeName())
        self.type_name = '(' + ', '.join(tparams) + ')'

    def has_children(self):
        return self.valobj.MightHaveChildren()

    def num_children(self):
        return self.valobj.GetNumChildren()

    def get_child_at_index(self, index):
        child = self.valobj.GetChildAtIndex(index)
        return child.CreateChildAtOffset(str(index), 0, child.GetType())

    def get_index_of_child(self, name):
        return str(name)

    def get_summary(self):
        return tuple_summary(self.valobj)

    def get_type_name(self):
        return self.type_name


class MsvcEnumSynthProvider(EnumSynthProvider):
    is_tuple_variant = False

    def update(self):
        tparams = get_template_params(self.valobj.GetTypeName())
        if len(tparams) == 1:  # Regular enum
            discr = gcm(self.valobj, 'discriminant')
            self.variant = gcm(self.valobj, 'variant' + str(discr.GetValueAsUnsigned()))
            variant_name = discr.GetValue()
        else:  # Niche enum
            dataful_min = int(tparams[1])
            dataful_max = int(tparams[2])
            dataful_var = tparams[3]
            discr = gcm(self.valobj, 'discriminant')
            if dataful_min <= discr.GetValueAsUnsigned() <= dataful_max:
                self.variant = gcm(self.valobj, 'dataful_variant')
                variant_name = dataful_var
            else:
                variant_name = discr.GetValue()

        self.type_name = tparams[0]

        if self.variant.IsValid() and self.variant.GetNumChildren() > self.skip_first:
            if self.variant.GetChildAtIndex(self.skip_first).GetName() == '__0':
                self.is_tuple_variant = True
                self.summary = variant_name + tuple_summary(self.variant, skip_first=self.skip_first)
            else:
                self.summary = variant_name + '{...}'
        else:
            self.summary = variant_name

    def get_child_at_index(self, index):
        child = self.variant.GetChildAtIndex(index + self.skip_first)
        if self.is_tuple_variant:
            return child.CreateChildAtOffset(str(index), 0, child.GetType())
        else:
            return child

    def get_index_of_child(self, name):
        if self.is_tuple_variant:
            return int(name)
        else:
            return self.variant.GetIndexOfChildWithName(name) - self.skip_first

    def get_type_name(self):
        return self.type_name


class MsvcEnum2SynthProvider(EnumSynthProvider):
    is_tuple_variant = False

    def update(self):
        tparams = get_template_params(self.valobj.GetTypeName())
        self.type_name = tparams[0]

    def has_children(self):
        return True

    def get_child_at_index(self, index):
        return self.valobj.GetChildAtIndex(index)

    def get_index_of_child(self, name):
        return self.valobj.GetChildIndex(name)

    def get_type_name(self):
        return self.type_name


##################################################################################################################


class StdHashMapSynthProvider(RustSynthProvider):
    def update(self):
        self.initialize_table(gcm(self.valobj, 'base', 'table'))

    def initialize_table(self, table):
        assert table.IsValid()

        if table.type.GetNumberOfTemplateArguments() > 0:
            item_ty = table.type.GetTemplateArgumentType(0)
        else:  # we must be on windows-msvc - try to look up item type by name
            table_ty_name = table.GetType().GetName()  # "hashbrown::raw::RawTable<ITEM_TY>"
            item_ty_name = get_template_params(table_ty_name)[0]
            item_ty = table.GetTarget().FindTypes(item_ty_name).GetTypeAtIndex(0)

        if item_ty.IsTypedefType():
            item_ty = item_ty.GetTypedefedType()

        inner_table = table.GetChildMemberWithName('table')
        if inner_table.IsValid():
            self.initialize_hashbrown_v2(inner_table, item_ty)  # 1.52 <= std_version
        else:
            if not table.GetChildMemberWithName('data'):
                self.initialize_hashbrown_v2(table, item_ty)  # ? <= std_version < 1.52
            else:
                self.initialize_hashbrown_v1(table, item_ty)  # 1.36 <= std_version < ?

    def initialize_hashbrown_v2(self, table, item_ty):
        self.num_buckets = gcm(table, 'bucket_mask').GetValueAsUnsigned() + 1
        ctrl_ptr = gcm(table, 'ctrl', 'pointer')
        ctrl = ctrl_ptr.GetPointeeData(0, self.num_buckets)
        # Buckets are located above `ctrl`, in reverse order.
        start_addr = ctrl_ptr.GetValueAsUnsigned() - item_ty.GetByteSize() * self.num_buckets
        buckets_ty = item_ty.GetArrayType(self.num_buckets)
        self.buckets = self.valobj.CreateValueFromAddress('data', start_addr, buckets_ty)
        error = lldb.SBError()
        self.valid_indices = []
        for i in range(self.num_buckets):
            if ctrl.GetUnsignedInt8(error, i) & 0x80 == 0:
                self.valid_indices.append(self.num_buckets - 1 - i)

    def initialize_hashbrown_v1(self, table, item_ty):
        self.num_buckets = gcm(table, 'bucket_mask').GetValueAsUnsigned() + 1
        ctrl_ptr = gcm(table, 'ctrl', 'pointer')
        ctrl = ctrl_ptr.GetPointeeData(0, self.num_buckets)
        buckets_ty = item_ty.GetArrayType(self.num_buckets)
        self.buckets = gcm(table, 'data', 'pointer').Dereference().Cast(buckets_ty)
        error = lldb.SBError()
        self.valid_indices = []
        for i in range(self.num_buckets):
            if ctrl.GetUnsignedInt8(error, i) & 0x80 == 0:
                self.valid_indices.append(i)

    def has_children(self):
        return True

    def num_children(self):
        return len(self.valid_indices)

    def get_child_at_index(self, index):
        bucket_idx = self.valid_indices[index]
        item = self.buckets.GetChildAtIndex(bucket_idx)
        return item.CreateChildAtOffset('[%d]' % index, 0, item.GetType())

    def get_index_of_child(self, name):
        return int(name.lstrip('[').rstrip(']'))

    def get_summary(self):
        return 'size=%d, capacity=%d' % (self.num_children(), self.num_buckets)


class StdHashSetSynthProvider(StdHashMapSynthProvider):
    def update(self):
        table = gcm(self.valobj, 'base', 'map', 'table')  # std_version >= 1.48
        if not table.IsValid():
            table = gcm(self.valobj, 'map', 'base', 'table')  # std_version < 1.48
        self.initialize_table(table)

    def get_child_at_index(self, index):
        bucket_idx = self.valid_indices[index]
        item = self.buckets.GetChildAtIndex(bucket_idx).GetChildAtIndex(0)
        return item.CreateChildAtOffset('[%d]' % index, 0, item.GetType())

##################################################################################################################


def __lldb_init_module(debugger_obj, internal_dict): # pyright: ignore
    LOG.warn('Initializing...')
    initialize_category(debugger_obj, internal_dict)