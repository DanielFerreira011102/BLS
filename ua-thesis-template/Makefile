# Name of your main tex file (without .tex extension)
MAIN = main
# Output directory for compiled files
OUT_DIR = build

# Compiler options
LATEXMK = latexmk
LATEXMK_OPTIONS = -lualatex -output-directory=$(OUT_DIR)

# Default target: Compile the LaTeX document
all: pdf

# Generate the PDF
pdf:
	$(LATEXMK) $(LATEXMK_OPTIONS) $(MAIN).tex

# View the generated PDF (adjust if you use a different PDF viewer)
view: pdf
	open $(OUT_DIR)/$(MAIN).pdf

# Clean temporary files, keeping the final PDF
clean:
	$(LATEXMK) -c -output-directory=$(OUT_DIR)

# Full clean: Remove all generated files including the PDF
cleanall:
	$(LATEXMK) -C -output-directory=$(OUT_DIR)
	rm -rf $(OUT_DIR)

# Run a continuous preview (auto-rebuild on file changes)
watch:
	$(LATEXMK) $(LATEXMK_OPTIONS) -pvc $(MAIN).tex

.PHONY: all pdf view clean cleanall watch

