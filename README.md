This is an interpreter for expressions written in the Lambda Calculus notation.

- Uses Python to lex, parse, and evaluate expressions written in untyped Lambda Calculus
- As of now, does not enable function definition by name, but coming soon
- As of now, Alpha-reduction is implemented with assigning novel numbers to overloaded variables, enabling technical accuracy, but intuitive and, well, better variable naming under Alpha-reduction is coming soon
- Does not use lambda-Python built in functionality (on purpose!)
  - instead, purely typographically reduces/evaluates well-formed Lambda Calculus expressions