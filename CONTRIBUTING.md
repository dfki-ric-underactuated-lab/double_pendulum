# How to Contribute

Contributions to this project are very welcome!

If you discover bugs, have feature requests, or want to improve the
documentation, please open an issue at the issue tracker of the project.

If you want to contribute code, please open a pull request via GitHub by
forking the project, committing changes to your fork, and then opening a pull
request from your forked branch to the main branch.

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

Please try to follow the software guidelines from this repository as described
in the documentation.

If you contribute a controller, please

1. Create a .rst file in the docs/text folder in which you explain how the
   controller works. If possible add references.
2. Write meaningful docstrings for your code (numpy doctring format).
3. If your controller requires additional libraries, add them to
   src/python/setup.py (for python libraries) or add them to the installation
   guide.
