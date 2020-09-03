# roi-gan

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vayOHR71rL1OOizqo7G2SvGrCaoyvT6t)

![Generation over canvas](http://jcboyd.github.io/assets/roi-gan/canvas.gif)

<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

An RoI discriminator for image-to-image GANs,

<img src="https://render.githubusercontent.com/render/math?math=d_{roi} : X \times B \to \{0, 1\}">

for image domain <img src="https://render.githubusercontent.com/render/math?math=X"> and bounding box domain <img src="https://render.githubusercontent.com/render/math?math=B"> such that,

<img src="https://render.githubusercontent.com/render/math?math=d_{roi}(\mathbf{x}, b) = d(\rho(f(\mathbf{x}), b))">

for feature <img src="https://render.githubusercontent.com/render/math?math=f"> and discrimination <img src="https://render.githubusercontent.com/render/math?math=d"> layers, separated by RoIAlign layer <img src="https://render.githubusercontent.com/render/math?math=\rho">. This quantises RoIs into a fixed size,

<img src="https://render.githubusercontent.com/render/math?math=\rho : X \times B \to \mathbf{y},">

for input <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x} \in \mathbb{R}^{1\times W \times H \times C},"> <img src="https://render.githubusercontent.com/render/math?math=k"> bounding box tuples <img src="https://render.githubusercontent.com/render/math?math=b">, and output <img src="https://render.githubusercontent.com/render/math?math=\mathbf{y} \in \mathbb{R}^{k \times w \times h \times C}"> with quantised dimensions <img src="https://render.githubusercontent.com/render/math?math=w < W"> and <img src="https://render.githubusercontent.com/render/math?math=h < H">.

<!-- \begin{figure}
\centering
  \includestandalone[width=0.5\textwidth]{tikz/roigan}%     without .tex extension
  % or use \input{mytikz}
%  \caption{Fluorescence labeler, based on U-Net architecture \cite{ronneberger2015u}.}
  \label{fig:hca_pipeline}
\end{figure} -->

Akin to the PatchGAN, the discriminator output averages over the individual bounding boxes,

<img src="https://render.githubusercontent.com/render/math?math=D_{roi} = \frac{1}{|\mathcal{B}(\mathbf{x})|}\sum_{b \in \mathcal{B}(\mathbf{x})} d_{roi}(\mathbf{x}, b),">

where the operator <img src="https://render.githubusercontent.com/render/math?math=\mathcal{B}"> returns the set of bounding boxes of image <img src="https://render.githubusercontent.com/render/math?math=\mathbf{x}">.
