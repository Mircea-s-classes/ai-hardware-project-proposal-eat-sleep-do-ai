from tvm import relax
from mlc_llm import interface

mod, params = interface.load_model("SmolLM2-1.7B-Instruct-q4f16_1-MLC")
mod.show()  # 打印 Relax IR

