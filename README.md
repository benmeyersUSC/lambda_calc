This is an interpreter for expressions written in the Lambda Calculus notation.

- Uses Python to lex, parse, and evaluate expressions written in untyped Lambda Calculus
- Does not use lambda-Python built in functionality (on purpose!)
  - instead, purely typographically reduces/evaluates well-formed Lambda Calculus expressions


Functionality
- Takes in a .lambda (.txt) file which contains assignments or expressions separated by line breaks
- Prints final reduced expressions as well as parsed syntax trees along the way
- Also saves files containing text-visualized syntax trees with and without assignment statements plugged in
- Allows for assignment names with lowercase letters, uppercase letters (if not first character!), no numbers or other symbols at all