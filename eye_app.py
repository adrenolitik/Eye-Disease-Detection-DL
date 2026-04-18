from fastai.vision.all import *
import gradio as gr

# Compatibility shim for older pickled Learner objects that expect
# `plum.resolver.Resolver.warn_redefinition` to exist.
try:
    from plum.resolver import Resolver
    if not hasattr(Resolver, "warn_redefinition"):
        Resolver.warn_redefinition = False
except Exception:
    pass

# Some legacy pickles store `plum.function.MethodType` as a serializable class.
# In modern plum it maps to builtin `method`, so we temporarily patch only
# while deserializing and restore right away for runtime compatibility.
try:
    import plum.function as _plum_function

    class _CompatMethodType:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["state"] = state

    _original_method_type = _plum_function.MethodType
    _plum_function.MethodType = _CompatMethodType
    try:
        learn = load_learner('eye_disease_model.pkl')
    finally:
        _plum_function.MethodType = _original_method_type
except Exception:
    learn = load_learner('eye_disease_model.pkl')

def predict_eye_disease(img):
    # Ensure compatibility with the learner's expected fastai image type.
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
