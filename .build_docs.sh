python -V
echo "Removing previously built documentation files..."
rm -rf ./docs/build/
echo "Building documentations..."
#sphinx-apidoc -f -o ./docs/source/ ./pyxc/ --separate --module-first
cd ./docs
echo "Compiling documentations..."
make html