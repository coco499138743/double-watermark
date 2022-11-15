# Double watermarking
Abstract: Image processing networks have gained great success inmany fields,
and thus the issue of copyright protection for image processing networks has
become a focus of attention. Model watermarking techniques are widely used
in model copyright protection, but there are two challenges: (1) designing
universal trigger sample watermarking for different network models is still
a challenge; (2) existing methods of copyright protection based on trigger s
watermarking are difficult to resist forgery attacks. In this work, we propose a
dual model watermarking framework for copyright protection in image processing
networks. The trigger sample watermark is embedded in the training
process of the model, which can effectively verify the model copyright.And we
design a common method for generating trigger sample watermarks based on
generative adversarial networks, adaptively generating trigger sample watermarks
according to different models. The spatial watermark is embedded into
the model output. When an attacker steals model copyright using a forged
trigger sample watermark, which can be correctly extracted to distinguish
between the piratical and the protected model. The experiments show that the
proposed framework has good performance in different image segmentation
networks of UNET, UNET++, and FCN (fully convolutional network), and
effectively resists forgery attacks.
