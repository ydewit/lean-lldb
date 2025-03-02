// Lean compiler output
// Module: Main
// Imports: Init
#include <lean/lean.h>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C"
{
#endif
    lean_object *_id_func(lean_object *x){
        return x;
    }

    LEAN_EXPORT lean_object *_lean_main(lean_object *);
    LEAN_EXPORT lean_object *_lean_main(lean_object *x_1)
    {
    _start:
    {
        lean_object *boxed_scalar_0 = lean_box(0);
        lean_object *boxed_scalar_99 = lean_box(99);
        lean_object *boxed_scalar_MAX = lean_box(SIZE_MAX);

        // LeanConstructor
        lean_object *ctor = lean_alloc_ctor(0, 3, 0);
        lean_ctor_set(ctor, 0, lean_box(3));
        lean_ctor_set(ctor, 1, lean_box(6));
        lean_ctor_set(ctor, 2, lean_box(9));

        // LeanClosure
        lean_object *closure = lean_alloc_closure((void*)(_id_func), 2, 1);
        lean_closure_set(closure, 0, lean_box(33));

        // LeanArray
        // lean_object *arr = lean_mk_empty_array();
        lean_object *arr = lean_mk_empty_array_with_capacity(lean_box(3));
        // if ( arr->m_tag == LeanArray ){
            lean_array_push(arr, lean_box(2));
            lean_array_push(arr, lean_box(4));
            lean_array_push(arr, lean_box(6));

            // LeanString
            lean_object *str = lean_mk_string("A test string");
        // }

        lean_inc(closure);
        lean_object* thunk_0 = lean_mk_thunk(closure);

        lean_object* thunk_1 = lean_thunk_pure(lean_box(4));

        return lean_io_result_mk_ok(lean_box(0));
    }
    }
    
    lean_object *initialize_Init(uint8_t builtin, lean_object *);
    static bool _G_initialized = false;
    LEAN_EXPORT lean_object *initialize_Main(uint8_t builtin, lean_object *w)
    {
        lean_object *res;
        if (_G_initialized)
            return lean_io_result_mk_ok(lean_box(0));
        _G_initialized = true;
        res = initialize_Init(builtin, lean_io_mk_world());
        if (lean_io_result_is_error(res))
            return res;
        lean_dec_ref(res);
        return lean_io_result_mk_ok(lean_box(0));
    }
    void lean_initialize_runtime_module();

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#endif

    int main(int argc, char **argv)
    {
#if defined(WIN32) || defined(_WIN32)
        SetErrorMode(SEM_FAILCRITICALERRORS);
#endif
        lean_object *in;
        lean_object *res;
        lean_initialize_runtime_module();
        lean_set_panic_messages(false);
        res = initialize_Main(1 /* builtin */, lean_io_mk_world());
        lean_set_panic_messages(true);
        lean_io_mark_end_initialization();
        if (lean_io_result_is_ok(res))
        {
            lean_dec_ref(res);
            lean_init_task_manager();
            res = _lean_main(lean_io_mk_world());
        }
        lean_finalize_task_manager();
        if (lean_io_result_is_ok(res))
        {
            int ret = 0;
            lean_dec_ref(res);
            return ret;
        }
        else
        {
            lean_io_result_show_error(res);
            lean_dec_ref(res);
            return 1;
        }
    }
#ifdef __cplusplus
}
#endif
