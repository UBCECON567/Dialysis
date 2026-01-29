using Dialysis
using Documenter

mathengine = MathJax3(Dict(
    #:loader => Dict("load" => ["[tex]/physics"]),
    :tex => Dict(
        "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
        "displayMath" => [["\$\$", "\$\$"], ["\\[", "\\]"]
    ],

        "tags" => "ams",
        "packages" => ["base", "ams", "autoload"],
    ),
    :equationNumbers => Dict(:autoNumber => "AMS"),
    :Macros => Dict(
        :ind => ["\\mathbbm{1}_{#1}\\left(#2\\right)", 2],
        :R => ["\\mathbbm{R}"]
    ),
))

# render qmd files
function replace_math_blocks(input_file::String, output_file::String)    content = read(input_file, String)
    modified_content = replace(content, r"\$\$(.*?)\$\$"s => s"""```math\n\1\n```""")
    # Write the modified content to the output file
    write(output_file, modified_content)
end

function replace_img(input_file::String, output_file::String)
    content = read(input_file, String)
    modified_content = replace(content, r"<img.*?\nsrc=\"(.*?\.png)\"\n.*?>"s => s"![](\1)")
    #println(modified_content)
    # Write the modified content to the output file
    write(output_file, modified_content)
end

function replace_questions(s)
    # Pattern to match the entire note/question block
    pattern = r"> \[!NOTE\]\s*>?\s*>\s*### Question\s*((?:>.*\n?)+)"
    return replace(s, pattern => s -> begin
                       # Extract the quoted block
                       m = match(pattern, s)
                       quoted = m.captures[1]
                       # Replace each '> ' line with indented text
                       indented = replace(quoted, r"^>(.*)"m => s"    \1")
                       # Compose the final block
                       "!!! question\n$indented"
                   end)
end

function replace_obsidian_tip_with_admonition(s)
    # Pattern to match the entire tip block
    pattern = r"> \[!TIP\]\s*>?\s*>\s*### (.*)\s*>?\s*((?:>.*\n?)+)"
    return replace(s, pattern => s -> begin
          m = match(pattern, s)
          title = m.captures[1]
          quoted = m.captures[2]
          indented = replace(quoted, r"^>(.*)"m => s"    \1")
          "!!! tip \"$title\"\n\n$indented"
      end)
end


function replace_admonitions(infile, outfile)
    s = read(infile, String)
    # Replace the note/question header
    s = replace_questions(s)
    s = replace_obsidian_tip_with_admonition(s)
    write(outfile,s)
end



Base.cd(@__DIR__) # change to the directory of this script
input_dir = "qmd"
output_dir= "src"
qmd_files = filter(f -> endswith(f, ".qmd") && !startswith(f,"."), readdir(input_dir))
force_render=false
for qmd_file in qmd_files
    input_file = joinpath(input_dir, qmd_file)
    output_file = joinpath(output_dir, replace(basename(qmd_file), ".qmd" => ".md"))
    if (stat(input_file).mtime > stat(output_file).mtime) || force_render
        run(`quarto render $input_file --to gfm --output-dir ../$output_dir`)
        replace_math_blocks(output_file, output_file)
        replace_img(output_file, output_file)
        replace_admonitions(output_file, output_file)
    else
        @warn "$input_file not modified since last render, skipping"
    end
end


gitorg = "UBCECON567"
repo = "Dialysis"


makedocs(;
    modules=[Dialysis],
    authors="Paul Schrimpf <paul.schrimpf@ubc.ca> and contributors",
    repo="https://github.com/$gitorg/$repo/blob/{commit}{path}#{line}",
    sitename="$repo",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        mathengine=mathengine,
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Function Reference" => "index.md",
        "Assignment Part 1" => "dialysis-1.md",
        "Assignment Part 2" => "dialysis-2.md",
        "License" => "license.md",
    ],
         warnonly=true,
)

module Local

import Documenter

struct LocalConfig <: Documenter.DeployConfig
end

function Documenter.deploy_folder(cfg::LocalConfig; repo, devbranch, push_preview, devurl,
    tag_prefix, kwargs...)
    Documenter.DeployDecision(true, "gh-pages", false, repo, "")
end

function Documenter.authentication_method(cfg::LocalConfig)
    Documenter.SSH
end

## obtain
function Documenter.documenter_key(cfg::LocalConfig)
    @show read("ssh.key", String)
end

end



deploydocs(;
    repo="github.com/$gitorg/$repo.git", #Remotes.GitHub("ubcecon", "PartitioendRegression"),
    devbranch="master",
    target="build",
    deploy_config=Local.LocalConfig(),
    dirname="",
)
