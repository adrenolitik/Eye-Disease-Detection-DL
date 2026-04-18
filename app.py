from fastai.vision.all import *
import gradio as gr
import sys
import types


def _install_plum_compat_shims():
    """Provide legacy plum.* module paths expected by old pickles."""
    try:
        import plum._alias as _alias
        import plum._autoreload as _autoreload
        import plum._dispatcher as _dispatcher
        import plum._function as _function
        import plum._method as _method
        import plum._parametric as _parametric
        import plum._promotion as _promotion
        import plum._resolver as _resolver
        import plum._signature as _signature
        import plum._type as _type
        import plum._util as _util
    except Exception:
        return None, None

    class _CompatMethodType:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["state"] = state

    mappings = {
        "plum.alias": _alias,
        "plum.autoreload": _autoreload,
        "plum.dispatcher": _dispatcher,
        "plum.method": _method,
        "plum.parametric": _parametric,
        "plum.promotion": _promotion,
        "plum.resolver": _resolver,
        "plum.signature": _signature,
        "plum.type": _type,
        "plum.util": _util,
    }
    for module_name, module_obj in mappings.items():
        shim = types.ModuleType(module_name)
        shim.__dict__.update(module_obj.__dict__)
        sys.modules[module_name] = shim

    function_shim = types.ModuleType("plum.function")
    function_shim.__dict__.update(_function.__dict__)
    function_shim.MethodType = _CompatMethodType
    sys.modules["plum.function"] = function_shim

    resolver_shim = sys.modules["plum.resolver"]
    if hasattr(resolver_shim, "Resolver") and not hasattr(resolver_shim.Resolver, "warn_redefinition"):
        resolver_shim.Resolver.warn_redefinition = False

    original_method_type = getattr(_function, "MethodType", None)
    _function.MethodType = _CompatMethodType
    return _function, original_method_type


def _load_learner_compat(path):
    patched_module, original_method_type = _install_plum_compat_shims()
    try:
        return load_learner(path)
    finally:
        if patched_module is not None and original_method_type is not None:
            patched_module.MethodType = original_method_type


learn = _load_learner_compat("eye_disease_model.pkl")


def predict_eye_disease(img):
    pred, pred_idx, probs = learn.predict(PILImage.create(img))
    return {str(pred): float(probs[pred_idx])}


interface = gr.Interface(
    fn=predict_eye_disease,
    inputs=gr.Image(
        type="pil",
        label="Размести фото здесь - или - нажми и загрузи",
    ),
    outputs=gr.Label(num_top_classes=1, label="Результат анализа"),
    title="ИИ диагностика болезней глаз",
    description="Загрузите фото сетчатки для ИИ диагностики катаракты, глаукомы, диабетической ретинопатии и др.",
    submit_btn="Анализировать",
    clear_btn="Очистить",
    flagging_mode="manual",
    css="""
footer { display: none !important; }
""",
    js="""
() => {
  const dictionary = {
    "Flag": "Записать",
    "Drop Image Here": "Размести фото здесь",
    "Click to Upload": "Нажми и загрузи",
    "or": "или",
    "Submit": "Анализировать",
    "Clear": "Очистить",
    "Use via API": "Использовать через API"
  };

  const localizeTextNodes = () => {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
      const raw = node.nodeValue;
      if (raw) {
        const trimmed = raw.trim();
        if (dictionary[trimmed]) {
          node.nodeValue = raw.replace(trimmed, dictionary[trimmed]);
        }
      }
      node = walker.nextNode();
    }
  };

  const replaceText = () => {
    document.querySelectorAll("button").forEach((btn) => {
      const text = btn.textContent ? btn.textContent.trim() : "";
      if (dictionary[text]) {
        btn.textContent = dictionary[text];
      }
    });
    localizeTextNodes();
  };

  replaceText();
  new MutationObserver(replaceText).observe(document.body, { childList: true, subtree: true });
}
""",
)

interface.launch()
