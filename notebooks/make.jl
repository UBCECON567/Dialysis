using Weave
weave("dialysis.jmd",out_path=:doc,
      mod=Main,
      doctype="pandoc2html",
      pandoc_options=["--toc","--toc-depth=2","--filter=pandoc-citeproc"],
      cache=:user) 
notebook("dialysis.jmd", :pwd, -1, "--allow-errors")
