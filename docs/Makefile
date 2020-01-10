BIB = jmd/dialysis.bib
PANDOC = pandoc --bibliography $(BIB) -f markdown -t markdown_mmd-citations --metadata link-citations=true 

WEAVE = julia makeweave.jl 
vpath %.jmd jmd

jmdin := $(wildcard jmd/*.jmd)
jmdmd := $(patsubst jmd/%.jmd, md/%.md,$(jmdin))
jmdout := $(patsubst jmd/%.jmd, build/%.md,$(jmdin))

docin := $(wildcard src/*.md)
docout := $(patsubst src/%.md, build/%.md,$(docin))


mdout := $(jmdout) 

nbout := $(patsubst jmd/%.jmd, build/%.ipynb,$(jmdin))


#build/figure : figure
#	ln -s figure build/figure

build/figures : md/figures
	ln -s md/figures build/figures

jmd : $(jmdout) 
nb : $(nbout)
doc: $(docout)

$(jmdmd) : md/%.md : %.jmd #makeweave.jl 
	$(WEAVE) $< $@  weave

$(mdout) : build/%.md : md/%.md
	@mkdir -p $(@D)
	$(PANDOC) $< -o $@

$(nbout) : build/%.ipynb : %.jmd
	@mkdir -p $(@D)
	$(WEAVE) $< $@  notebook

$(docout) : build/%.md :  src/%.md
	julia -e "using Pkg; Pkg.activate(\".\"); using Documenter, Dialysis, DocumenterMarkdown; makedocs( modules=[Dialysis],  format=Markdown(), clean=false,  pages=[  \"Home\" => \"index.md\" ], repo=\"https://github.com/UBCECON567/Dialysis/blob/{commit}{path}#L{line}\",  sitename=\"Reproducing Grieco & McDevitt (2017)\",  authors=\"Paul Schrimpf <schrimpf@mail.ubc.ca>\")"

site: site/index.html

site/index.html: $(jmdout) $(docout) mkdocs.yml
	mkdocs build

all: site/index.html $(nbout)

deploy: site/index.html 
	mkdocs gh-deploy
