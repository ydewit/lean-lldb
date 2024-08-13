from __future__ import print_function, division
import sys
import logging
import lldb
import weakref

if sys.version_info[0] == 2:
    # python2-based LLDB accepts utf8-encoded ascii strings only.
    def to_lldb_str(s): return s.encode('utf8', 'backslashreplace') if isinstance(s, unicode) else s
    range = xrange
else:
    to_lldb_str = str

LOG = logging.getLogger(__name__)
MODULE = sys.modules[__name__]
CATEGORY = None

MAX_STRING_SUMMARY_LENGTH = 1024


def initialize_category(debugger, internal_dict):
    global MODULE
    global CATEGORY

    CATEGORY = debugger.CreateCategory('Lean')
    CATEGORY.SetEnabled(True)

    attach_synthetic_to_type(LeanSynthProvider, 'lean_object')
    attach_synthetic_to_type(LeanSynthProvider, 'lean_object*')

    # attach_summary_to_type(tuple_summary_provider, r'^\(.*\)$', True)
    # attach_synthetic_to_type(MsvcTupleSynthProvider, r'^tuple\$?<.+>$', True)  # *-windows-msvc uses this name since 1.47

    # attach_synthetic_to_type(StrSliceSynthProvider, '&str')
    # attach_synthetic_to_type(StrSliceSynthProvider, 'str*')
    # attach_synthetic_to_type(StrSliceSynthProvider, 'str')  # *-windows-msvc uses this name since 1.5?
    # attach_synthetic_to_type(StrSliceSynthProvider, 'ref$<str$>')
    # attach_synthetic_to_type(StrSliceSynthProvider, 'ref_mut$<str$>')

    # attach_synthetic_to_type(StdStringSynthProvider, '^(collections|alloc)::string::String$', True)
    # attach_synthetic_to_type(StdVectorSynthProvider, r'^(collections|alloc)::vec::Vec<.+>$', True)
    # attach_synthetic_to_type(StdVecDequeSynthProvider,
    #                          r'^(collections|alloc::collections)::vec_deque::VecDeque<.+>$', True)

    # attach_synthetic_to_type(MsvcEnumSynthProvider, r'^enum\$<.+>$', True)
    # attach_synthetic_to_type(MsvcEnum2SynthProvider, r'^enum2\$<.+>$', True)

    # attach_synthetic_to_type(SliceSynthProvider, r'^&(mut *)?\[.*\]$', True)
    # attach_synthetic_to_type(MsvcSliceSynthProvider, r'^(mut *)?slice\$?<.+>.*$', True)
    # attach_synthetic_to_type(MsvcSliceSynthProvider, r'^ref(_mut)?\$<slice2\$<.+>.*>$', True)

    # attach_synthetic_to_type(StdCStringSynthProvider, '^(std|alloc)::ffi::c_str::CString$', True)
    # attach_synthetic_to_type(StdCStrSynthProvider, '^&?(std|core)::ffi::c_str::CStr$', True)
    # attach_synthetic_to_type(StdCStrSynthProvider, 'ref$<core::ffi::c_str::CStr>')
    # attach_synthetic_to_type(StdCStrSynthProvider, 'ref_mut$<core::ffi::c_str::CStr>')

    # attach_synthetic_to_type(StdOsStringSynthProvider, 'std::ffi::os_str::OsString')
    # attach_synthetic_to_type(StdOsStrSynthProvider, '^&?std::ffi::os_str::OsStr', True)
    # attach_synthetic_to_type(StdOsStrSynthProvider, 'ref$<std::ffi::os_str::OsStr>')
    # attach_synthetic_to_type(StdOsStrSynthProvider, 'ref_mut$<std::ffi::os_str::OsStr>')

    # attach_synthetic_to_type(StdPathBufSynthProvider, 'std::path::PathBuf')
    # attach_synthetic_to_type(StdPathSynthProvider, '^&?std::path::Path', True)
    # attach_synthetic_to_type(StdPathSynthProvider, 'ref$<std::path::Path>')
    # attach_synthetic_to_type(StdPathSynthProvider, 'ref_mut$<std::path::Path>')

    # attach_synthetic_to_type(StdRcSynthProvider, r'^alloc::rc::Rc<.+>$', True)
    # attach_synthetic_to_type(StdRcSynthProvider, r'^alloc::rc::Weak<.+>$', True)
    # attach_synthetic_to_type(StdArcSynthProvider, r'^alloc::(sync|arc)::Arc<.+>$', True)
    # attach_synthetic_to_type(StdArcSynthProvider, r'^alloc::(sync|arc)::Weak<.+>$', True)
    # attach_synthetic_to_type(StdMutexSynthProvider, r'^std::sync::mutex::Mutex<.+>$', True)

    # attach_synthetic_to_type(StdCellSynthProvider, r'^core::cell::Cell<.+>$', True)
    # attach_synthetic_to_type(StdRefCellSynthProvider, r'^core::cell::RefCell<.+>$', True)
    # attach_synthetic_to_type(StdRefCellBorrowSynthProvider, r'^core::cell::Ref<.+>$', True)
    # attach_synthetic_to_type(StdRefCellBorrowSynthProvider, r'^core::cell::RefMut<.+>$', True)

    # attach_synthetic_to_type(StdHashMapSynthProvider, r'^std::collections::hash::map::HashMap<.+>$', True)
    # attach_synthetic_to_type(StdHashSetSynthProvider, r'^std::collections::hash::set::HashSet<.+>$', True)

    # attach_synthetic_to_type(GenericEnumSynthProvider, r'^core::option::Option<.+>$', True)
    # attach_synthetic_to_type(GenericEnumSynthProvider, r'^core::result::Result<.+>$', True)
    # attach_synthetic_to_type(GenericEnumSynthProvider, r'^alloc::borrow::Cow<.+>$', True)

    if 'lean' in internal_dict.get('source_languages', []):
        lldb.SBDebugger.SetInternalVariable('target.process.thread.step-avoid-regexp',
                                            '^<?(std|core|alloc)::', debugger.GetInstanceName())

    MAX_STRING_SUMMARY_LENGTH = debugger.GetSetting('target.max-string-summary-length').GetIntegerValue()


def attach_synthetic_to_type(synth_class, type_name, is_regex=False):
    global module, CATEGORY
    # log.debug('attaching synthetic %s to "%s", is_regex=%s', synth_class.__name__, type_name, is_regex)
    synth = lldb.SBTypeSynthetic.CreateWithClassName(__name__ + '.' + synth_class.__name__)
    synth.SetOptions(lldb.eTypeOptionCascade)
    CATEGORY.AddTypeSynthetic(lldb.SBTypeNameSpecifier(type_name, is_regex), synth)

    def summary_fn(valobj, dict): return get_synth_summary(synth_class, valobj, dict)
    # LLDB accesses summary fn's by name, so we need to create a unique one.
    summary_fn.__name__ = '_get_synth_summary_' + synth_class.__name__
    setattr(module, summary_fn.__name__, summary_fn)
    attach_summary_to_type(summary_fn, type_name, is_regex)


def attach_summary_to_type(summary_fn, type_name, is_regex=False):
    global module, CATEGORY
    # log.debug('attaching summary %s to "%s", is_regex=%s', summary_fn.__name__, type_name, is_regex)
    summary = lldb.SBTypeSummary.CreateWithFunctionName(__name__ + '.' + summary_fn.__name__)
    summary.SetOptions(lldb.eTypeOptionCascade)
    CATEGORY.AddTypeSummary(lldb.SBTypeNameSpecifier(type_name, is_regex), summary)


# 'get_summary' is annoyingly not a part of the standard LLDB synth provider API.
# This trick allows us to share data extraction logic between synth providers and their sibling summary providers.
def get_synth_summary(synth_class, valobj, dict):
    try:
        obj_id = valobj.GetIndexOfChildWithName('$$object-id$$')
        summary = LeanObjectLikeSynthProvider.synth_by_id[obj_id].get_summary()
        return to_lldb_str(summary)
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


# ----- Synth providers ------

def is_scalar(o):
    return o.unsigned & 1 == 1

def box_scalar(n):
    return lldb.SBValue.CreateValueFromExpression(None, None, '((lean_object*)(%s << 1 | 1))' % n)

def unbox_scalar(o):
    return o.unsigned >> 1

def get_tag(o):
    return o.GetChildMemberWithName('m_tag').unsigned

def get_num_objs(o):
    return o.GetChildMemberWithName('m_other').unsigned

def get_rc(o):
    return o.GetChildMemberWithName('m_rc').unsigned

def get_ctor_obj(o, i):
    return o.Cast(lldb.target.FindFirstType('lean_ctor_object').GetPointerType()).GetChildMemberWithName('m_objs').GetChildAtIndex(i)


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

def call_function(name, returns, *args):
    argStrs = [ ]
    for arg in args:
        if isinstance(arg, str):
            argStrs.append('"%s"' % arg.replace('\\', '\\\\').replace('"', '\"'))
        elif isinstance(arg, lldb.SBValue):
            argStrs.append(arg.GetName())
        else:
            argStrs.append(repr(arg))
    expr = '(%s) %s(%s)' % (returns, name, ', '.join(argStrs))
    return lldb.SBValue.CreateValueFromExpression(None, None, expr)

# typedef struct {
#     int      m_rc;
#     unsigned m_cs_sz:16;
#     unsigned m_other:8;
#     unsigned m_tag:8;
# } lean_object;
class LeanObjectLikeSynthProvider(object):
    synth_by_id = weakref.WeakValueDictionary()
    next_id = 0

    def __init__(self, valobj, ptr_type = None, dict={}):
        self.obj_id = type(self).next_id
        self.valobj = valobj
        type(self).synth_by_id[self.obj_id] = self
        type(self).next_id += 1

    def addr_size(self):
        return self.valobj.GetTarget().GetAddressByteSize()

    def _call(self, name, returns, *args):
        argStrs = [ ]
        for arg in args:
            if isinstance(arg, str):
                argStrs.append('"%s"' % arg.replace('\\', '\\\\').replace('"', '\"'))
            elif isinstance(arg, lldb.SBValue):
                argStrs.append(arg.GetName())
            else:
                argStrs.append(repr(arg))
        expr = '(%s)%s(%s)' % (returns, name, ', '.join(argStrs))
        return self.valobj.CreateValueFromExpression(name, expr)

    def size(self):
        return self._call('lean_object_byte_size', 'size_t', self.valobj).GetValueAsUnsigned()

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
        return None

    def get_summary(self):
        pass


class LeanBoxedScalarSynthProvider(LeanObjectLikeSynthProvider):
    def get_summary(self):
        return "(boxed scalar size=%d) %d" % (unbox_scalar(self.valobj), self.size())

# typedef struct {
#     lean_object   m_header;
#     lean_object * m_objs[0];
# } lean_ctor_object;

class LeanCtorSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_ctor_object').GetPointerType()), dict)
        self.tag = get_tag(self.valobj)
        self.rc = get_rc(self.valobj)
        self.num_objs = get_num_objs(self.valobj)
        # NOTE: ctor objects do not store scalar size, so we can't get it
        # self.num_scalars = self.valobj.GetChildMemberWithName('m_cs_sz').unsigned(self.valobj)

    def has_children(self):
        if self.num_objs > 0:
            return True
        return False
    
    def has_scalars(self):
        return (self.size() - self.addr_size() - self.num_objs * self.addr_size()) > 0
    
    def num_children(self):
        return self.num_objs
    
    def get_child_at_index(self, index):
        return get_ctor_obj(self.valobj, index)

    def get_summary(self):
        return "(Ctor#%u rc=%s num_objs=%u size=%s has_scalars=%s)" % (self.tag, str(self.rc) if self.rc != 0 else "∞", self.num_objs, self.size(), "true" if self.has_scalars() else "false")


# typedef struct {
#     lean_object   m_header;
#     void *        m_fun;
#     uint16_t      m_arity;     /* Number of arguments expected by m_fun. */
#     uint16_t      m_num_fixed; /* Number of arguments that have been already fixed. */
#     lean_object * m_objs[0];
# } lean_closure_object;
class LeanClosureSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_closure_object').GetPointerType()), dict)
        self.fun = self.valobj.GetChildMemberWithName('m_fun')
        self.arity = self.valobj.GetChildMemberWithName('m_arity').GetValueAsUnsigned()
        self.num_fixed = self.valobj.GetChildMemberWithName('m_num_fixed').GetValueAsUnsigned()
        self.objs = self.valobj.GetChildMemberWithName('m_objs')

    def has_children(self):
        return self.num_objs > 0

    def num_children(self):
        return self.num_objs
    
    def get_child_at_index(self, index):
        return self.objs.GetChildAtIndex(index).GetSummary()
    
    def get_summary(self):
        return "(Closure fun=%s arity=%s num_fixed=%u)" % (hex(self.fun.GetValueAsAddress()), self.arity, self.num_fixed)
    
# /* Array arrays */
# typedef struct {
#     lean_object   m_header;
#     size_t        m_size;
#     size_t        m_capacity;
#     lean_object * m_data[0];
# } lean_array_object;
class LeanArraySynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_array_object').GetPointerType()), dict)
        self.size = self.valobj.GetChildMemberWithName('m_size').GetValueAsUnsigned()
        self.capacity = self.valobj.GetChildMemberWithName('m_capacity').GetValueAsUnsigned()
        self.data = self.valobj.GetChildMemberWithName('m_data')

    def has_children(self):
        return self.size > 0
    
    def num_children(self):
        return self.size
    
    def get_child_at_index(self, index):
        return self.data.GetChildAtIndex(index).GetSummary()
    
    def get_summary(self):
        return "(Array size=%u capacity=%u)" % (self.size, self.capacity)

class LeanStructArraySynthProvider(LeanObjectLikeSynthProvider):
    def get_summary(self):
        return "(StructArray)"
    

# /* Scalar arrays */
# typedef struct {
#     lean_object   m_header;
#     size_t        m_size;
#     size_t        m_capacity;
#     uint8_t       m_data[0];
# } lean_sarray_object;
class LeanScalarArraySynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.GetTarget().FindFirstType('lean_sarray_object').GetPointerType(), dict)
        self.size = self.valobj.GetChildMemberWithName('m_size').GetValueAsUnsigned()
        self.capacity = self.valobj.GetChildMemberWithName('m_capacity').GetValueAsUnsigned()
        self.data = self.valobj.GetChildMemberWithName('m_data')

    def has_children(self):
        return self.size > 0
    
    def num_children(self):
        return self.size
    
    def get_child_at_index(self, index):
        return box_scalar(self.data.GetChildAtIndex(index).unsigned)
    
    def get_summary(self):
        return "(ScalarArray size=%u capacity=%u)" % (self.size, self.capacity)
    
# typedef struct {
#     lean_object m_header;
#     size_t      m_size;     /* byte length including '\0' terminator */
#     size_t      m_capacity;
#     size_t      m_length;   /* UTF8 length */
#     char        m_data[0];
# } lean_string_object;
class LeanStringSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_string_object').GetPointerType()), dict)
        self.size = self.valobj.GetChildMemberWithName('m_size').GetValueAsUnsigned()
        self.capacity = self.valobj.GetChildMemberWithName('m_capacity').GetValueAsUnsigned()
        self.length = self.valobj.GetChildMemberWithName('m_length').GetValueAsUnsigned()
        self.data = self.valobj.GetChildMemberWithName('m_data').Cast(self.valobj.GetTarget().FindFirstType('char').GetPointerType()).GetSummary()
        #.Cast(self.valobj.GetTarget().FindFirstType('char').GetPointerType()).GetValue()

    def get_summary(self):
        strval = string_from_ptr(self.data, min(self.length, MAX_STRING_SUMMARY_LENGTH))
        if self.length > MAX_STRING_SUMMARY_LENGTH:
            strval += u'...'
        strVal = u'"%s"' % strval
        return "(String size=%u capacity=%u lenght=%u %s)" % (self.size, self.capacity, self.length, strVal)

class LeanMpzSynthProvider(LeanObjectLikeSynthProvider):
    def get_summary(self):
        return "(MPZ)"
    
# typedef struct {
#     lean_object            m_header;
#     _Atomic(lean_object *) m_value;
#     _Atomic(lean_object *) m_closure;
# } lean_thunk_object;
class LeanThunkSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_thunk_object').GetPointerType()), dict)
        self.value = self.valobj.GetChildMemberWithName('m_value')
        self.closure = self.valobj.GetChildMemberWithName('m_closure')

    def get_summary(self):
        return "(Thunk value=%s closure=%s)" % (self.value.GetSummary(), self.closure.GetSummary())
    
# typedef struct lean_task {
#     lean_object            m_header;
#     _Atomic(lean_object *) m_value;
#     lean_task_imp *        m_imp;
# } lean_task_object;
class LeanTaskSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_task_object').GetPointerType()), dict)
        self.value = self.valobj.GetChildMemberWithName('m_value')
        self.imp = self.valobj.GetChildMemberWithName('m_imp')

    def get_summary(self):
        return "(Task value=%s)" % self.value.GetSummary()
    
# typedef struct {
#     lean_object   m_header;
#     lean_object * m_value;
# } lean_ref_object;
class LeanRefSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_ref_object').GetPointerType()), dict)
        self.value = self.valobj.GetChildMemberWithName('m_value')

    def get_summary(self):
        return "(Ref value=%s)" % self.value.GetSummary()


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
class LeanExternalSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        super().__init__(valobj.Cast(valobj.GetTarget().FindFirstType('lean_external_object').GetPointerType()), dict)
        self.class_ = self.valobj.GetChildMemberWithName('m_class')
        self.data = self.valobj.GetChildMemberWithName('m_data')

    def get_summary(self):
        return "(External class=%s data=%s)" % (self.class_.GetSummary(), self.data.GetSummary())


class LeanReservedSynthProvider(LeanObjectLikeSynthProvider):
    def __init__ (self, valobj, dict={}):
        pass

    def get_summary(self):
        return "(reserved)"

##################################################################################################################


class LeanSynthProvider(object):
    def __init__(self, valobj, dict={}):
        if is_scalar(valobj):
            self.provider = LeanBoxedScalarSynthProvider(valobj)
        else:
            tag = get_tag(valobj)
            if tag <= LEAN_MAX_CTOR_TAG:
                self.provider = LeanCtorSynthProvider(valobj)
            elif tag == LEAN_CLOSURE:
                self.provider = LeanClosureSynthProvider(valobj)
            elif tag == LEAN_ARRAY:
                self.provider = LeanArraySynthProvider(valobj)
            elif tag == LEAN_STRUCT_ARRAY:
                self.provider = LeanStructArraySynthProvider(valobj)
            elif tag == LEAN_SCALAR_ARRAY:
                self.provider = LeanScalarArraySynthProvider(valobj)
            elif tag == LEAN_STRING:
                self.provider = LeanStringSynthProvider(valobj)
            elif tag == LEAN_MPZ:
                self.provider = LeanMpzSynthProvider(valobj)
            elif tag == LEAN_THUNK:
                self.provider = LeanThunkSynthProvider(valobj)
            elif tag == LEAN_TASK:
                self.provider = LeanTaskSynthProvider(valobj)
            elif tag == LEAN_REF:
                self.provider = LeanRefSynthProvider(valobj)
            elif tag == LEAN_EXTERNAL:
                self.provider = LeanExternalSynthProvider(valobj)
            elif tag == LEAN_RESERVED:
                self.provider = LeanReservedSynthProvider(valobj)
            else:
                raise Exception('Unknown lean object tag: %d' % tag)

    def update(self):
        return self.provider.update()

    def has_children(self):
        return self.provider.has_children()

    def num_children(self):
        return self.provider.num_children()

    def get_child_at_index(self, index):
        return self.provider.get_child_at_index(index)

    def get_child_index(self, name):
        return self.provider.get_child_index(name)

    def get_summary(self):
        return self.provider.get_summary()


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