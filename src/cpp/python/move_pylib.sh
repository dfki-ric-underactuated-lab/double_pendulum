if [ $(python3 -c 'import site; print(site.ENABLE_USER_SITE)') == True ]
then
    mv cppilqr.cpython-*.so $(python3 -c 'import site; print(site.USER_SITE+"/")')
else
    mv cppilqr.cpython-*.so $(python3 -c 'import site; print(site.getsitepackages()[0]+"/")')
fi
