# Scientific limitations

The released CAU ImF pairs do not include the original steganographic secret key, owner message, and decoder. Therefore exact target match and teacher-forced target NLL are Level-A characterization metrics, not native ImF payload verification success. Native payload fields remain null.

AWQ3 is saved as dequantized quantized-on-grid Hugging Face weights; it is not described as an upstream packed INT3 runtime. AWQ/GPTQ require the exact completed IF calibration artifact. RTN requires the exact backend from that same completed experiment; the pipeline exits explicitly when either dependency is unavailable.

