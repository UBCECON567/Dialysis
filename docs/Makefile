BIB = jmd/dialysis.bib
PANDOC = pandoc --bibliography $(BIB) -f markdown -t markdown_mmd-citations --metadata link-citations=true 

WEAVE = julia makeweave.jl 
vpath %.jmd jmd

jmdin := $(wildcard jmd/*.jmd)
jmdmd := $(patsubst jmd/%.jmd, md/%.md,$(jmdin))
jmdout := $(patsubst jmd/%.jmd, src/%.md,$(jmdin))

#docin := $(wildcard src/*.md)
#docout := $(patsubst src/%.md, build/%.md,$(docin))


mdout := $(jmdout) 

nbout := $(patsubst jmd/%.jmd, src/%.ipynb,$(jmdin))


#build/figure : figure
#	ln -s figure build/figure

#build/figures : md/figures
#	ln -s md/figres jmd/figures

jmd : $(jmdout) 
nb : $(nbout)
doc: $(docout)

$(rmdmd) : md/%.md : %.Rmd
	$(RKNIT) $< $@

$(jmdmd) : md/%.md : %.jmd #makeweave.jl 
	$(WEAVE) $< $@  weave

$(mdout) : src/%.md : md/%.md
	$(PANDOC) $< -o $@

$(nbout) : src/%.ipynb : %.jmd 
	$(WEAVE) $< $@  notebook

$(docout) : build/%.md :  src/%.md
	julia -e "using Documenter, NeuralNetworkEconomics, DocumenterMarkdown; makedocs( modules=[NeuralNetworkEconomics],  format=Markdown(), clean=false,  pages=[  \"Home\" => \"index.md\" ], repo=\"https://github.com/UBCECON567/Dialysis/blob/{commit}{path}#L{line}\",  sitename=\"Dialysis\",  authors=\"Paul Schrimpf <paul.schrimpf@gmail.com>\")"

#site: site/index.html

#site/index.html: $(rmdout) $(jmdout) $(docout) mkdocs.yml
#	mkdocs build

#all: site/index.html $(nbout)

#deploy: site/index.html 
#	mkdocs gh-deploy
