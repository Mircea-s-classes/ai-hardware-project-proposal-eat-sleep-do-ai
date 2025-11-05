### **Towards Fast and Efficient Deep Neural Network Deployment: Supporting Diverse Quantization in the TVM Compiler**

#### **Background**

As neural networks continue to scale up—with billions of parameters becoming common—the computational and memory costs of model deployment increase dramatically. Quantization has therefore emerged as one of the primary techniques for efficient deep neural network inference, enabling both model compression and faster execution. To deploy quantized models on various hardware platforms, machine learning compilers such as TVM play a crucial role in transforming high-level computation graphs into optimized low-level code for CPUs, GPUs, and accelerators. Effective compiler-level support for quantization is thus essential to realize the theoretical efficiency gains in practice.

#### **Challenges**

Although quantization techniques have advanced significantly, several challenges remain for compiler-level deployment:
(i) **Limited datatype, bit-width, and granularity support:** Existing compilers often struggle to handle heterogeneous quantization schemes—spanning multiple datatypes (e.g., INT, Floating-point, Posit) , precision(e.g., INT4, INT8, E4M3) and granularities (e.g., per-channel, per-group). This limitation can lead to sub-optimal kernel selection, reduced operator fusion, and degraded inference efficiency.
(ii) **Hardware–datatype mismatch:** Customized quantization formats may not align well with the arithmetic units of general-purpose processors, causing the actual inference latency to diverge from theoretical expectations. For instance, converting FP32 weights to INT16 may not yield noticeable speedups without proper compiler and hardware co-optimization.
(iii) **Expanding** **optimization** **design space:** As modern networks grow deeper and adopt mixed-precision quantization across layers, the compiler’s auto-tuner must navigate a larger design space. Each layer introduces multiple candidate configurations—including quantization bit-widths, datatypes, and scheduling parameters—resulting in high computational overhead and tuning cycles. Efficiently managing this complexity remains a challenge.

#### **Methods**

To address these challenges, we propose an **end-to-end TVM-based quantized neural network compilation framework** with the following components:
(i) **Quantization-aware transformation pass:** Integrate datatype and bit-width information directly into the intermediate representation (IR) to enable more precise operator fusion, datatype propagation, and kernel optimization for quantized models.
(ii) **Efficient Compilation Flow Auto-Tuner**: Incorporate hierarchical search and prior-guided optimization to efficiently explore the enlarged design space introduced by large-scale models and mixed quantization. This approach improves tuning efficiency and reduces compilation time.
(iii) **End-to-end deployment integration:** Develop a unified quantization workflow that bridges model import, IR transformation, autotuning, and runtime deployment within TVM. This enables consistent quantization metadata handling and streamlined optimization across all compilation stages, providing a end-to-end solution for quantized model deployment.

#### **Expected Outcome**

The proposed framework will enable flexible and efficient deployment of quantized deep neural networks across heterogeneous hardware backends, achieving lower inference latency, higher throughput, and scalable autotuning performance for large models.