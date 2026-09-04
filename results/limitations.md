# Scientific limitations

The historical released CAU ImF pairs do not include their original steganographic key, owner message, and decoder; their existing results remain Level-A exact-match/NLL characterization only. The new `data/imf_generated` experiment does not claim to recover those artifacts: it creates a separate ADG instance with a committed key/message and reports native decoded-payload FSR only for checkpoints trained from those new pairs.

AWQ3 is saved as dequantized quantized-on-grid Hugging Face weights; it is not described as an upstream packed INT3 runtime. AWQ/GPTQ require the exact completed IF calibration artifact. RTN requires the exact backend from that same completed experiment; the pipeline exits explicitly when either dependency is unavailable.
