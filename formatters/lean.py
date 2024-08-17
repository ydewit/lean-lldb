from __future__ import print_function, division
import sys
import logging
import lldb
import weakref

LOG = logging.getLogger(__name__)

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

MAX_STRING_SUMMARY_LENGTH = 512

def __lldb_init_module(debugger, _dict):  # pyright: ignore
    global MAX_STRING_SUMMARY_LENGTH

    LOG.warning("Initializing...")

    category = debugger.CreateCategory("Lean")

    attach_synthetic_to_type(category, LeanSynthProvider, 'lean_object *')

    MAX_STRING_SUMMARY_LENGTH = debugger.GetSetting(
        "target.max-string-summary-length"
    ).GetIntegerValue()

    print_category(category)

    category.SetEnabled(True)

def attach_synthetic_to_type(category, synth_class, type_name, is_regex=False):
    LOG.info('attaching synthetic %s to "%s", is_regex=%s', synth_class.__name__, type_name, is_regex)

    synth = lldb.SBTypeSynthetic.CreateWithClassName(__name__ + "." + synth_class.__name__)
    synth.SetOptions(lldb.eTypeOptionCascade)

    category.AddTypeSynthetic(lldb.SBTypeNameSpecifier(type_name, is_regex), synth)

    def summary_fn(valobj, dictionary):
        return get_synth_summary(synth_class, valobj, dictionary)

    # LLDB accesses summary fn's by name, so we need to create a unique one.
    summary_fn.__name__ = "_get_synth_summary_" + synth_class.__name__
    setattr(sys.modules[__name__], summary_fn.__name__, summary_fn)
    attach_summary_to_type(category, summary_fn, type_name, is_regex)

def attach_summary_to_type(category, summary_fn, type_name, is_regex=False):
    LOG.info('attaching summary %s to "%s", is_regex=%s', summary_fn.__name__, type_name, is_regex)
    summary = lldb.SBTypeSummary.CreateWithFunctionName(__name__ + "." + summary_fn.__name__)
    summary.SetOptions(lldb.eTypeOptionCascade)
    category.AddTypeSummary(lldb.SBTypeNameSpecifier(type_name, is_regex), summary)

def print_category(category):
    for cat_id in range(category.GetNumFormats()):
        LOG.info('Format: %s', category.GetFormatAtIndex(cat_id))
    for cat_id in range(category.GetNumSummaries()):
        LOG.info('Summary: %s', category.GetSummaryAtIndex(cat_id))
    for cat_id in range(category.GetNumFilters()):
        LOG.info('Filter: %s', category.GetFilterAtIndex(cat_id))
    for cat_id in range(category.GetNumSynthetics()):
        LOG.info('Synth: %s', category.GetSyntheticAtIndex(cat_id))

# 'get_summary' is annoyingly not a part of the standard LLDB synth provider API.
# This trick allows us to share data extraction logic between synth providers and their sibling summary providers.
def get_synth_summary(_synth_class, valobj, _dict):
    LOG.info('get_synth_summary %s', valobj)
    try:
        obj_id = valobj.GetIndexOfChildWithName("$$object-id$$")
        summary = LeanSynthProvider.get_synth_by_id(obj_id).get_summary()
        return str(summary)
    except KeyError as e:
        LOG.exception("%s", e)
    except Exception as e:
        LOG.exception("%s", e)
        raise


# Chained GetChildMemberWithName lookups
def gcm(valobj, *chain):
    for name in chain:
        valobj = valobj.GetChildMemberWithName(name)
    return valobj


# Get a pointer out of core::ptr::Unique<T>
def read_unique_ptr(valobj):
    pointer = valobj.GetChildMemberWithName("pointer")
    if pointer.TypeIsPointerType():  # Between 1.33 and 1.63 pointer was just *const T
        return pointer
    return pointer.GetChildAtIndex(0)


def string_from_ptr(pointer, length):
    if length <= 0:
        return ""
    error = lldb.SBError()
    process = pointer.GetProcess()
    data = process.ReadMemory(pointer.GetValueAsUnsigned(), length, error)
    if error.Success():
        return data.decode("utf8", "replace")
    else:
        raise Exception("ReadMemory error: %s", error.GetCString())


def obj_summary(valobj, unavailable="{...}"):
    summary = valobj.GetSummary()
    if summary is not None:
        return summary
    summary = valobj.GetValue()
    if summary is not None:
        return summary
    return unavailable


def sequence_summary(childern, maxsize=32):
    s = ""
    for child in childern:
        if len(s) > 0:
            s += ", "
        s += obj_summary(child)
        if len(s) > maxsize:
            s += ", ..."
            break
    return s


def tuple_summary(obj, skip_first=0):
    fields = [
        obj_summary(obj.GetChildAtIndex(i))
        for i in range(skip_first, obj.GetNumChildren())
    ]
    return f'({", ".join(fields)})'


# ----- Summaries -----


def tuple_summary_provider(valobj, _dict={}):
    return tuple_summary(valobj)

def downcast_and_deref_ptr(valobj, type_name):
    """
    Downcasts a pointer and dereferences it.

    This is used to cast a 'lean_object*' to a specific subtype,
    e.g. 'lean_ctor_object'.

    Args:
        valobj (SBValue): The pointer value object.
        type_name (str): The name of the type to downcast to.

    Returns:
        SBValue: The downcasted and dereferenced value object.
    """
    assert valobj.GetType().IsPointerType(), 'must be a pointer type'
    sbtype = valobj.GetTarget().FindFirstType(type_name)
    if sbtype:
        # for some reason `lean_string_object` is not found by `FindFirstType`
        # in this case, we need manualy read the fields with memory offsets
        valobj = valobj.Cast(sbtype.GetPointerType())
    return valobj.Dereference()

def assert_ptr_type(valobj):
    """
    Asserts that the given value object is a pointer type.

    Args:
        valobj: The value object to be checked.

    Raises:
        AssertionError: If the value object is not a pointer type.
    """
    assert valobj.GetType().IsPointerType(), 'must be a pointer type'

def assert_lean_object_ptr(valobj):
    """
    Asserts that the given value object is a pointer to a `lean_object`.

    Args:
        valobj: The value object to be checked.

    Raises:
        AssertionError: If the value object is not a pointer type or not a `lean_object`.
    """
    assert valobj.GetType().IsPointerType(), 'must be a pointer type'
    assert valobj.Dereference().GetType().GetName() == 'lean_object', 'not a lean_object'

# ----- Synth providers ------
class LikeSynthProvider(object):
    """
    typedef struct {
        int      m_rc;
        unsigned m_cs_sz:16;
        unsigned m_other:8;
        unsigned m_tag:8;
    } lean_object;

    """

    def __init__(self, lean_value):
        # lean_value can be a pointer to a lean_opject or one of its subtype
        # It will be a pointer to lean_object if LLDB can't find the sybtype
        self.m_lean_value = lean_value

    def to_summary(self, kind, summary):
        rc = str(self.get_rc()) if self.get_rc() != 0 else "∞"
        return '(%s.{%s} %s)' % (kind, rc, summary)
    
    def get_total_size(self):
        expr = f"(size_t)lean_object_byte_size((lean_object *){self.get_lean_value().AddressOf()})"
        size = self.get_lean_value().GetTarget().EvaluateExpression(expr)
        return size.GetValueAsUnsigned(0)

    def get_lean_value(self):
        return self.m_lean_value

    def is_subtype(self):
        return self.get_lean_value().GetType().GetName() != 'lean_object'

    def get_header_type(self):
        if self.is_subtype():
            return self.get_lean_value().GetTarget().FindFirstType('lean_object')
        else:
            return self.get_lean_value().GetType()
        
    def get_header_value(self):
        if self.is_subtype():
            return self.get_lean_value().GetChildMemberWithName("m_header")
        else:
            return self.get_lean_value()

    def get_child_value(self, child_name, child_offset, child_type):
        if self.is_subtype():
            return self.get_lean_value().GetChildMemberWithName(child_name)
        else:
            return self.get_lean_value().CreateChildAtOffset(child_name, child_offset, child_type)

    def get_tag(self):
        return self.get_header_value().GetChildMemberWithName("m_tag").GetValueAsUnsigned()

    def get_other(self):
        return self.get_header_value().GetChildMemberWithName("m_other").GetValueAsUnsigned()

    def get_rc(self):
        return self.get_header_value().GetChildMemberWithName("m_rc").GetValueAsSigned()

    def get_address_size(self):
        return self.get_lean_value().GetTarget().GetAddressByteSize()

    def get_target(self):
        return self.get_lean_value().GetTarget()

    def update(self):
        return True

    def has_children(self):
        return False

    def num_children(self):
        return 0

    def get_child_at_index(self, index):
        return None

    def get_child_index(self, name):
        return -1

    def get_summary(self):
        return "<not implemented>"

class LeanSynthProvider(object):

    @staticmethod
    def cast(valobj, type_name):
        assert valobj.GetType().IsPointerType(), 'valobj must be a pointer type'
        return valobj.Cast(valobj.GetTarget().FindFirstType(type_name).GetPointerType())

    _synth_by_id = weakref.WeakValueDictionary()
    _next_id = 0

    @staticmethod
    def _get_next_id():
        obj_id = LeanSynthProvider._next_id
        LeanSynthProvider._next_id += 1
        return obj_id

    @staticmethod
    def get_synth_by_id(obj_id):
        provider = LeanSynthProvider._synth_by_id[obj_id]
        return provider

    @staticmethod
    def _add_synth_by_id(provider):
        LeanSynthProvider._synth_by_id[provider.m_obj_id] = provider
        return provider.m_obj_id

    @staticmethod
    def is_scalar(valobj):
        return valobj.GetType().IsPointerType() and valobj.GetValueAsUnsigned(0) & 1 == 1

    def __init__(self, lean_object, _dict):
        assert_ptr_type(lean_object)
        LOG.info('LEAN_OBJECT: (%s)%s - %s', lean_object.GetType(), lean_object.GetName(), lean_object)
        
        if LeanSynthProvider.is_scalar(lean_object):
            self.m_provider = LeanBoxedScalarProvider(lean_object)
        else:
            tag_value = lean_object.Dereference().GetChildMemberWithName("m_tag")
            tag = tag_value.GetValueAsUnsigned()
            if tag <= LEAN_MAX_CTOR_TAG:
                self.m_provider = LeanCtorProvider(lean_object)
            elif tag == LEAN_CLOSURE:
                self.m_provider = LeanClosureProvider(lean_object)
            elif tag == LEAN_ARRAY:
                self.m_provider = LeanArrayProvider(lean_object)
            elif tag == LEAN_STRUCT_ARRAY:
                self.m_provider = LeanStructArrayProvider(lean_object)
            elif tag == LEAN_SCALAR_ARRAY:
                self.m_provider = LeanScalarArrayProvider(lean_object)
            elif tag == LEAN_STRING:
                self.m_provider = LeanStringProvider(lean_object)
            elif tag == LEAN_MPZ:
                self.m_provider = LeanMpzProvider(lean_object)
            elif tag == LEAN_THUNK:
                self.m_provider = LeanThunkProvider(lean_object)
            elif tag == LEAN_TASK:
                self.m_provider = LeanTaskProvider(lean_object)
            elif tag == LEAN_REF:
                self.m_provider = LeanRefProvider(lean_object)
            elif tag == LEAN_EXTERNAL:
                self.m_provider = LeanExternalProvider(lean_object)
            elif tag == LEAN_RESERVED:
                self.m_provider = LeanReservedProvider(lean_object)
            else:
                raise Exception('Should never happen')

        self.m_obj_id = LeanSynthProvider._get_next_id()
        LeanSynthProvider._add_synth_by_id(self)

    def update(self):
        if self.m_provider is not None:
            return self.m_provider.update()
        else:
            LOG.error('|||| update')

    def has_children(self):
        if self.m_provider is not None:
            return self.m_provider.has_children()
        LOG.error('|||| has_children')

    def num_children(self):
        if self.m_provider is not None:
            return self.m_provider.num_children()
        LOG.error('|||| num_children')

    def get_child_at_index(self, index):
        if self.m_provider is not None:
            return self.m_provider.get_child_at_index(index)
        LOG.error('|||| get_child_at_index')

    def get_child_index(self, name):
        if name == "$$object-id$$":
            return self.m_obj_id
        if self.m_provider is not None:
            return self.m_provider.get_child_index(name)
        LOG.error('|||| get_child_index')

    def get_summary(self):
        if self.m_provider is not None:
            return self.m_provider.get_summary()
        LOG.error('|||| get_summary')


class DefaultSynthProvider(LikeSynthProvider):
    def __init__(self, objval):
        self.m_objval = objval

    def update(self):
        return self.m_objval.Update()

    def has_children(self):
        return self.m_objval.MightHaveChildren()

    def num_children(self):
        return self.m_objval.GetNumChildren()

    def get_child_at_index(self, index):
        return self.m_objval.GetChildAtIndex(index)

    def get_child_index(self, name):
        return self.m_objval.GetIndexOfChildWithName(name)

    def get_summary(self):
        return self.m_objval.GetSummary()


class LeanObjectProvider(LikeSynthProvider):
    VOID_PTR_TYPE = None
    UINT8_T_TYPE = None
    UINT16_T_TYPE = None
    SIZE_T_TYPE = None
    CHAR_TYPE = None
    UNSIGNED_TYPE = None
    INT_TYPE = None
    LEAN_OBJECT_TYPE = None

    @staticmethod
    def get_total_size(valobj):
        assert not valobj.GetType().IsPointerType(), "valobj is a pointer"
        expr = f"(size_t)lean_object_byte_size({valobj.AddressOf()})"
        size = valobj.GetTarget().EvaluateExpression(expr)
        return size.GetValueAsUnsigned(0)

    def __init__(self, obj):
        self.m_obj = obj

        # Initialize common static types (target may not be available otherwise)
        if LeanObjectProvider.VOID_PTR_TYPE is None:
            LeanObjectProvider.VOID_PTR_TYPE = (
                self.m_obj.GetTarget().FindFirstType("void").GetPointerType()
            )

        if LeanObjectProvider.UINT8_T_TYPE is None:
            LeanObjectProvider.UINT8_T_TYPE = (
                self.m_obj.GetTarget().FindFirstType("uint8_t")
            )

        if LeanObjectProvider.UINT16_T_TYPE is None:
            LeanObjectProvider.UINT16_T_TYPE = (
                self.m_obj.GetTarget().FindFirstType("uint16_t")
            )

        if LeanObjectProvider.SIZE_T_TYPE is None:
            LeanObjectProvider.SIZE_T_TYPE = self.m_obj.GetTarget().FindFirstType(
                "size_t"
            )

        if LeanObjectProvider.CHAR_TYPE is None:
            LeanObjectProvider.CHAR_TYPE = self.m_obj.GetTarget().FindFirstType(
                "char"
            )

        if LeanObjectProvider.UNSIGNED_TYPE is None:
            LeanObjectProvider.UNSIGNED_TYPE = (
                self.m_obj.GetTarget().FindFirstType("unsigned")
            )

        if LeanObjectProvider.INT_TYPE is None:
            LeanObjectProvider.INT_TYPE = self.m_obj.GetTarget().FindFirstType(
                "int"
            )

        if LeanObjectProvider.LEAN_OBJECT_TYPE is None:
            LeanObjectProvider.LEAN_OBJECT_TYPE = (
                self.m_obj.GetTarget().FindFirstType("lean_object")
            )

        self.update()

    # ----- lean_object fields ------

    def get_tag(self):  # unsigned
        return self.m_tag.GetValueAsUnsigned(0)

    def get_other(self):  # unsigned
        return self.m_other.GetValueAsUnsigned(0)

    def get_rc(self):  # signed
        return self.m_rc.GetValueAsSigned(0)

    def get_cs_sz(self):  # unsigned
        return self.m_cs_sz.GetValueAsUnsigned(0)

    def get_void_ptr_type(self):
        assert (
            LeanObjectProvider.VOID_PTR_TYPE is not None
        ), "void* type not initialized"
        return LeanObjectProvider.VOID_PTR_TYPE

    def get_uint8_t_type(self):
        assert (
            LeanObjectProvider.UINT8_T_TYPE is not None
        ), "uint8_t type not initialized"
        return LeanObjectProvider.UINT8_T_TYPE

    def get_uint16_t_type(self):
        assert (
            LeanObjectProvider.UINT16_T_TYPE is not None
        ), "uint16_t type not initialized"
        return LeanObjectProvider.UINT16_T_TYPE

    def get_size_t_type(self):
        assert (
            LeanObjectProvider.SIZE_T_TYPE is not None
        ), "size_t type not initialized"
        return LeanObjectProvider.SIZE_T_TYPE

    def get_char_type(self):
        assert (
            LeanObjectProvider.CHAR_TYPE is not None
        ), "char type bytes not initialized"
        return LeanObjectProvider.CHAR_TYPE

    def get_unsigned_type(self):
        assert (
            LeanObjectProvider.UNSIGNED_TYPE is not None
        ), "unsigned type not initialized"
        return LeanObjectProvider.UNSIGNED_TYPE

    def get_int_type(self):
        assert LeanObjectProvider.INT_TYPE is not None, "int type not initialized"
        return LeanObjectProvider.INT_TYPE

    def get_lean_object_type(self):
        assert (
            LeanObjectProvider.LEAN_OBJECT_TYPE is not None
        ), "lean_object type not initialized"
        return LeanObjectProvider.LEAN_OBJECT_TYPE

    def get_body_address_of(self):
        return (
            self.m_obj.GetValueAsUnsigned() + self.get_lean_object_type().GetByteSize()
        )

    def get_addr_size(self):  # uint8_t
        return self.m_obj.GetTarget().GetAddressByteSize()

    def get_type(self):  # SBType
        return self.m_obj.GetType()

    # def _call(self, name, returns, *args):
    #     argStrs = []
    #     for arg in args:
    #         if arg is None:
    #             raise Exception("None argument")
    #         elif arg.IsValid() == False:
    #             raise Exception("Invalid argument")
    #         elif isinstance(arg, str):
    #             LOG.warning("str argument type: %s", arg.GetValue())
    #             argStrs.append('"%s"' % arg.replace("\\", "\\\\").replace('"', '"'))
    #         else:
    #             argStrs.append(arg.GetValue())
    #     expr = f'({returns}){name}({", ".join(argStrs)})'
    #     return self.m_obj.GetFrame().EvaluateExpression(expr)

    # ----- Synth interface ------

    # update the internal state whenever the state of the variables in LLDB changes
    def update(self):
        self.m_rc = self.m_obj.GetChildMemberWithName("m_rc")
        self.m_cs_sz = self.m_obj.GetChildMemberWithName("m_cs_sz")
        self.m_other = self.m_obj.GetChildMemberWithName("m_other")
        self.m_tag = self.m_obj.GetChildMemberWithName("m_tag")
        return True

    def get_summary(self):
        return "{rc=%s,tag=%d,other=%u,cs_sz=%s}" % (
            self.get_rc() if self.get_rc() != 0 else "∞",
            self.get_tag(),
            self.get_other(),
            self.get_cs_sz(),
        )


class LeanBoxedScalarProvider(object):
    @staticmethod
    def box_scalar(boxed_scalar):
        return boxed_scalar.CreateValueFromExpression(
            None, f"((lean_object*)({boxed_scalar.GetValueAsUnsigned()} << 1 | 1))"
        )

    def __init__(self, lean_object):
        assert_ptr_type(lean_object)
        self.m_boxed_scalar = lean_object

    def get_scalar(self):
        return self.m_boxed_scalar.GetValueAsUnsigned() >> 1

    # ----- Synth interface ------
    def update(self):
        return True

    def has_children(self):
        return False

    def num_children(self):
        return 0

    def get_summary(self):
        return f"(Box {self.get_scalar()})"


class LeanCtorProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_object   m_header;
        lean_object * m_objs[0];
    } lean_ctor_object;

    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_ctor_object'))
        assert self.get_tag() <= LEAN_MAX_CTOR_TAG, 'invalid ctor idx'

    # ----- fields ------
    
    def get_ctor_idx(self):
        return self.get_tag()

    def get_num_objs(self):
        return self.get_other()

    def get_objs_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_objs")

    def has_scalars(self):
        """
        Check if the Lean object has scalars, since there is not enough
        metadata in the runtime object to determine this.

        Returns:
            bool: True if the Lean object has scalars, False otherwise.
        """
        return (
            # Call the Lean runtime to calculate the total object size
            self.get_total_size()
            - self.get_header_value().GetType().GetByteSize()
            - self.get_num_objs() * self.get_address_size()
            > 0
        )

    # ----- SynthProvider interface ------
 
    def update(self):
        return True

    def has_children(self):
        return self.get_num_objs() > 0

    def num_children(self):
        if self.get_num_objs() > LEAN_MAX_CTOR_FIELDS:
            return LEAN_MAX_CTOR_FIELDS
        return self.get_num_objs()

    def get_child_at_index(self, index):
        if index >= self.get_num_objs():
            return None
        if index < 0:
            return None

        offset = index * self.get_address_size()
        return self.get_objs_value().CreateChildAtOffset(f'[{index}]', offset, self.get_header_value().GetType().GetPointerType())

    def get_child_index(self, name):
        if name.startswith("[") and name.endswith("]"):
            return int(name.lstrip("[").rstrip("]"))
        return -1

    def get_summary(self):
        return self.to_summary(f'Ctor#{self.get_ctor_idx()}', f'objs={self.get_num_objs()}, scalars={str(self.has_scalars())}')

class LeanClosureProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_object   m_header;
        void *        m_fun;
        uint16_t      m_arity;     /* Number of arguments expected by m_fun. */
        uint16_t      m_num_fixed; /* Number of arguments that have been already fixed. */
        lean_object * m_objs[0];
    } lean_closure_object;

    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_closure_object'))
        self.update()

    # ----- fields ------

    def get_fun(self):  # SBFunction
        fun_address = self.get_lean_value().GetChildMemberWithName("m_fun").GetValueAsUnsigned()
        return self.get_target().ResolveLoadAddress(fun_address).GetFunction()

    def get_fun_decl(self):
        return self.m_fun_decl

    def get_fun_typename(self):
        return self.get_fun().GetType().GetName()

    def get_arity(self):  # unsigned
        arity = self.get_lean_value().GetChildMemberWithName("m_arity").GetValueAsUnsigned()
        return arity  - self.get_num_fixed()

    def get_num_fixed(self):  # unsigned
        return self.get_lean_value().GetChildMemberWithName("m_num_fixed").GetValueAsUnsigned()

    def get_objs_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_objs")

    # ----- SynthProvider interface ------

    def update(self):
        # update the computed function declaration
        return_type = self.get_fun().GetType().GetFunctionReturnType().GetName()
        arg_types = self.get_fun().GetType().GetFunctionArgumentTypes()
        self.m_fun_decl = f"{return_type} {self.get_fun().GetName()}({', '.join([arg.GetName() for arg in arg_types])})"
        return True

    def has_children(self):
        return self.get_num_fixed() > 0

    def num_children(self):
        return self.get_num_fixed()

    def get_child_at_index(self, index):
        if index >= self.get_num_fixed():
            return None
        if index < 0:
            return None
        offset = index * self.get_address_size()
        lean_object_ptr = self.get_header_value().GetType().GetPointerType()
        return self.get_objs_value().CreateChildAtOffset(f"[{index}]", offset, lean_object_ptr)

    def get_child_index(self, name):
        if name.startswith("[") and name.endswith("]"):
            return int(name.lstrip("[").rstrip("]"))
        return -1

    def get_summary(self):
        return self.to_summary('Clos', f'arity={self.get_arity()}, free={self.get_num_fixed()}, fun="{self.get_fun_decl()}"')


class LeanArrayProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_object   m_header;
        size_t        m_size;
        size_t        m_capacity;
        lean_object * m_data[0];
    } lean_array_object;

    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_array_object'))

    # ----- fields ------

    def get_size(self):  # unsigned
        return self.get_lean_value().GetChildMemberWithName("m_size").GetValueAsUnsigned()

    def get_capacity(self):  # unsigned
        return self.get_lean_value().GetChildMemberWithName("m_capacity").GetValueAsUnsigned()

    def get_data_value(self):  # SBValue
        return self.get_lean_value().GetChildMemberWithName("m_data")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def has_children(self):
        return self.get_size() > 0

    def num_children(self):
        return self.get_size()

    def get_child_at_index(self, index):
        if not 0 <= index < self.get_size():
            return None
        lean_object_ptr = self.get_header_value().GetType().GetPointerType()
        offset = lean_object_ptr.GetByteSize() * index
        return self.get_data_value().CreateChildAtOffset(f"[{index}]", offset, lean_object_ptr)

    def get_child_index(self, name):
        if name.startswith("[") and name.endswith("]"):
            return int(name.lstrip("[").rstrip("]"))
        return -1

    def get_summary(self):
        return self.to_summary('Arr', f'[{self.get_size()}/{self.get_capacity()}]')


class LeanStructArrayProvider(LikeSynthProvider):
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_struct_array_object'))

    def update(self):
        return True

    def get_summary(self):
        return "<not implemented>"

class LeanScalarArrayProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_object   m_header;
        size_t        m_size;
        size_t        m_capacity;
        uint8_t       m_data[0];
    } lean_sarray_object;
    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_sarray_object'))

    # ----- fields ------

    def get_size(self):
        return self.get_lean_value().GetChildMemberWithName("m_size").GetValueAsUnsigned()

    def get_capacity(self):
        return self.get_lean_value().GetChildMemberWithName("m_capacity").GetValueAsUnsigned()

    def get_data_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_data")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def has_children(self):
        return self.get_size() > 0

    def num_children(self):
        return self.get_size()

    def get_child_at_index(self, index):
        return LeanBoxedScalarProvider.box_scalar(
            self.get_data_value().GetChildAtIndex(index).GetValueAsUnsigned(0)
        )

    def get_summary(self):
        return self.to_summary('ScalarArr', f'size={self.get_size()}, capacity={self.get_capacity()}, ...')


class LeanStringProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_object m_header;
        size_t      m_size;     /* byte length including '\0' terminator */
        size_t      m_capacity;
        size_t      m_length;   /* UTF8 length */
        char        m_data[0];
    } lean_string_object;
    """
    def __init__(self, lean_object):
        # LOG.error('LeanStringProvider: %s', lean_object.GetTarget().FindFirstType('lean_string_object'))
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_string_object'))
        self.m_size_t_type = self.get_target().FindFirstType("size_t")
        self.m_char_ptr_type = self.get_target().FindFirstType("char").GetPointerType()

    # ----- fields ------

    def get_size(self):
        return self.get_child_value("m_size", self.get_header_type().GetByteSize(), self.m_size_t_type).GetValueAsUnsigned()

    def get_capacity(self):
        return self.get_child_value("m_capacity", self.get_header_type().GetByteSize() + self.m_size_t_type.GetByteSize(), self.m_size_t_type).GetValueAsUnsigned()

    def get_length(self):
        return self.get_child_value("m_size", self.get_header_type().GetByteSize() + self.m_size_t_type.GetByteSize() * 2, self.m_size_t_type).GetValueAsUnsigned()

    def get_data_value(self):
        return self.get_child_value("m_data", self.get_header_type().GetByteSize() + self.m_size_t_type.GetByteSize() * 3, self.m_char_ptr_type)

    def get_data_as_string(self):
        addr = self.get_data_value().AddressOf().GetValueAsUnsigned()
        if addr == 0:
            return ''
        error = lldb.SBError()
        content = self.get_data_value().process.ReadCStringFromMemory(
            addr, MAX_STRING_SUMMARY_LENGTH, error
        )
        if error.Success():
            if self.get_length() > MAX_STRING_SUMMARY_LENGTH:
                return f'{content}...'
            else:
                return f'{content}'
        else:
            return f'<error: {error.GetCString()}>'

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def has_children(self):
        return False

    def num_children(self):
        return 0

    def get_summary(self):
        LOG.error('LeanStringProvider: %s', self.get_other())
        return self.to_summary("Str", f'[{self.get_size()}/{self.get_capacity()}], len={self.get_length()}, "{self.get_data_as_string()}"')


class LeanMpzProvider(LikeSynthProvider):
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_mpz_object'))

    def update(self):
        return True

    def get_summary(self):
        return "<not implemented>"


# typedef struct {
#     lean_object            m_header;
#     _Atomic(lean_object *) m_value;
#     _Atomic(lean_object *) m_closure;
# } lean_thunk_object;
class LeanThunkProvider(LikeSynthProvider):
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_thunk_object'))
    # ----- fields ------

    def get_value_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_value")

    def get_closure_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_closure")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def get_summary(self):
        if self.get_value_value().GetValueAsUnsigned() == 0:
            return self.to_summary('Thunk', f'delayed={self.get_closure_value().GetSummary()}')
        else:
            return self.to_summary('Thunk', f'forced={self.get_value_value().GetSummary()}')


class LeanTaskProvider(LikeSynthProvider):
    """
    typedef struct lean_task {
        lean_object            m_header;
        _Atomic(lean_object *) m_value;
        lean_task_imp *        m_imp;
    } lean_task_object;
    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_task_object'))

    # ----- SynthProvider interface ------

    def get_value_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_value")

    def get_imp_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_imp")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def get_summary(self):
        if self.get_imp_value().GetValueAsUnsigned() == 0:
            return self.to_summary('Task', f'value: {self.get_value_value().GetSummary()}')
        else:
            return self.to_summary('Task', f'impl: {self.get_imp_value().GetSummary()}')

# typedef struct {
#     lean_object   m_header;
#     lean_object * m_value;
# } lean_ref_object;
class LeanRefProvider(LikeSynthProvider):
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_ref_object'))

    # fields

    def get_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_value")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def get_summary(self):
        return self.to_summary('Ref', 'value: %s' % self.get_value().GetSummary())


class LeanExternalProvider(LikeSynthProvider):
    """
    typedef struct {
        lean_external_finalize_proc m_finalize;
        lean_external_foreach_proc  m_foreach;
    } lean_external_class;
    
    /* Object for wrapping external data. */
    typedef struct {
        lean_object           m_header;
        lean_external_class * m_class;
        void *                m_data;
    } lean_external_object;
    """
    def __init__(self, lean_object):
        assert_lean_object_ptr(lean_object)
        super().__init__(downcast_and_deref_ptr(lean_object, 'lean_external_object'))

    # ----- fields ------

    def get_class_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_class")

    def get_data_value(self):
        return self.get_lean_value().GetChildMemberWithName("m_data")

    # ----- SynthProvider interface ------

    def update(self):
        return True

    def get_summary(self):
        return self.to_summary('Ext', f'class={self.get_class_value().GetSummary()}, data={self.get_data_value().GetSummary()}')


class LeanReservedProvider(LikeSynthProvider):
    """
    Represents a reserved lean_object

    :return: A string representing the summary of the LeanReservedProvider.
    """
    def __init__(self, lean_object):
        downcast_and_deref_ptr(lean_object, 'lean_object')
        super().__init__(lean_object)

    def get_summary(self):
        return self.to_summary('Reserved', 'not implemented')


class DebugSynthProvider(object):
    def __init__(self, lean_object, _dict):
        LOG.error('lean_object: %s , %s', lean_object.GetType(), lean_object.GetType())
        self.m_lean_object = lean_object

    def update(self):
        LOG.error('update')
        return True

    def has_children(self):
        LOG.error('has_children')
        return True
    
    def num_children(self):
        LOG.error('num_children')
        return 2
    
    def get_child_at_index(self, index):
        LOG.error('get_child_at_index %d', index)
        if index == 0:
            return self.m_lean_object.CreateValueFromExpression(f'{index}', '1')
        else:
            return self.m_lean_object.CreateValueFromExpression(f'{index}', '1')
    
    def get_child_index(self, name):
        LOG.error('get_child_index %s', name)
        return -1

    def get_summary(self):
        LOG.error('get_summary')
        return "LLLL"
