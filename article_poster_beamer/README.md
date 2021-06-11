Compiled with TeXLive 2020.

Compile the documents with

```powershell
latexmk -xelatex beamer1min beamer8min
latexmk poster hult_502 hult_502-supp arxiv
```

N.B. since the beamers uses the Metropolis theme, they need xelatex. But the other documents are processed by arXiv and AI, and they use (pdf)latex. So two calls are needed!

To upload to arXiv, create a tar archive with the following snippet (in powershell and using tarbsd)
N.B. you cannot do the archive with powershell `Compress-Archive`, since that used back-slash in the zip path names, which is invalid on mac/linux/unix, and arXiv rejects that. This is a bug. see https://superuser.com/questions/1382839/zip-files-expand-with-backslashes-on-linux-no-subdirectories

```powershell
$tar = "../submissions/arxiv_"+(get-date -format yyMMdd_hhmmss)+".tar.gz"
tar -czvf $tar arxiv.tex arxiv.bbl sections tikz data 
```
