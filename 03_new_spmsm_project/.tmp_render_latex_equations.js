"use strict";
const fs = require("fs");
const path = require("path");
const sharp = require("sharp");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");
const { AllPackages } = require("mathjax-full/js/input/tex/AllPackages.js");
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({packages: AllPackages});
const svgOut = new SVG({fontCache: "local"});
const doc = mathjax.document("", {InputJax: tex, OutputJax: svgOut});
const outDir = process.argv[2];
const equations = {
  "eq_6_1_targets.png": String.raw`T_{\mathrm{avg}}=\frac{1}{6}\sum_{k=1}^{6}T(\theta_k),\qquad \Delta T=\max_{1\le k\le 6}T(\theta_k)-\min_{1\le k\le 6}T(\theta_k)`,
  "eq_6_2_objectives.png": String.raw`\min_{\mathbf{x}\in\{0,1\}^{120}}\mathbf{f}(\mathbf{x})=\begin{bmatrix}f_1(\mathbf{x})\\f_2(\mathbf{x})\end{bmatrix}=\begin{bmatrix}-T_{\mathrm{avg}}(\mathbf{x})\\\Delta T(\mathbf{x})\end{bmatrix}`,
  "eq_6_3_mean.png": String.raw`\mu_t(\mathbf{x})=\frac{1}{4}\sum_{m=1}^{4}\widehat{y}_{m,t}(\mathbf{x}),\qquad t\in\{T_{\mathrm{avg}},\Delta T\}`,
  "eq_6_4_disagreement.png": String.raw`d_t(\mathbf{x})=\sqrt{\frac{1}{4}\sum_{m=1}^{4}\left[\widehat{y}_{m,t}(\mathbf{x})-\mu_t(\mathbf{x})\right]^2}`,
  "eq_6_5_combined.png": String.raw`D(\mathbf{x})=\sqrt{\frac{1}{2}\left[\left(\frac{d_{T}(\mathbf{x})}{s_{T}}\right)^2+\left(\frac{d_{\Delta T}(\mathbf{x})}{s_{\Delta T}}\right)^2\right]}`,
  "eq_7_1_cost.png": String.raw`N_{\mathrm{direct}}=5540\times6=33240,\quad N_{\mathrm{surrogate}}=601\times6=3606,\quad 1-\frac{N_{\mathrm{surrogate}}}{N_{\mathrm{direct}}}=89.15\%`,
  "eq_7_2_time.png": String.raw`t_{\mathrm{direct}}\approx\frac{5540\times33.40\ \mathrm{s}}{5}=10.28\ \mathrm{h},\quad t_{\mathrm{actual}}=1.58\ \mathrm{h},\quad \Delta t\approx8.70\ \mathrm{h}\ (84.6\%)`
};
async function render(name, latex) {
  let html = adaptor.outerHTML(doc.convert(latex, {display: true}));
  const a=html.indexOf("<svg"), b=html.indexOf("</svg>");
  let svg=html.slice(a,b+6).replace(/<\?xml[^>]*>/g,"").replace(/currentColor/g,"#000000");
  if(!/xmlns=/.test(svg)) svg=svg.replace(/<svg /,'<svg xmlns="http://www.w3.org/2000/svg" ');
  await sharp(Buffer.from(svg), {density: 450}).flatten({background:"white"}).png().toFile(path.join(outDir,name));
}
Promise.all(Object.entries(equations).map(([n,e])=>render(n,e))).catch(e=>{console.error(e);process.exit(1)});
