"""
- lexing (turning strings into named tokens we can work with)
- parsing (turn tokens into tree)
- interpret (take a tree and interpret directly)
"""
import re
import functools as func
def build_token_grabber(dct):
    return r'^(' + '|'.join([ f'(?P<{k}>{dct[k]})' for k in dct ]) + ')'

# Tokens for lexing
t_lambda = r'L'
t_left_paren = r'\('
t_right_paren = r'\)'
t_variable = r'[a-z]'
t_dot = r'\.'
t_whitespace = r'\s+'

token_dict = {
    't_lambda': r'L',
    't_left_paren': r'\(',
    't_right_paren': r'\)',
    't_variable': r'[a-z]',
    't_dot': r'\.',
    't_whitespace': r'\s+'
}

class Token:
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__

class LParen(Token):
    def __init__(self):
        super().__init__()

class RParen(Token):
    def __init__(self):
        super().__init__()

class Lambda(Token):
    def __init__(self):
        super().__init__()

class VarToken(Token):
    def __init__(self, val: str):
        super().__init__()
        self.val = val
    def __repr__(self):
        return super().__repr__() + f': {self.val}'

class Whitespace(Token):
    def __init__(self):
        super().__init__()

class Period(Token):
    def __init__(self):
        super().__init__()

def take_next_token(text: str):
    match_result = re.match(build_token_grabber(token_dict), text)
    result = None

    if match_result is None:
        raise Exception("Lambda invalid syntax")

    span = match_result.span()
    token = text[span[0]:span[1]]
    rest = text[span[1]:]

    dct = match_result.groupdict()
    for k in dct:
        mtch = dct[k]
        if mtch is None:
            continue

        if k == 't_lambda':
            result = Lambda()
        elif k == 't_left_paren':
            result = LParen()
        elif k == 't_right_paren':
            result = RParen()
        elif k == 't_variable':
            result = VarToken(token)
        elif k == 't_dot':
            result = Period()
        elif k == 't_whitespace':
            result = Whitespace()

    return (result, rest)

def tokenize(text: str):
    tokens_list = []

    while len(text) > 0:
        tok, rest = take_next_token(text)
        tokens_list.append(tok)
        text = rest
    return tokens_list



class Node:
    def __init__(self):
        pass

class Variable(Node):
    def __init__(self, variable_name: str):
        super().__init__()
        self.name = variable_name

    def __repr__(self):
        return self.name

class Abstraction(Node):
    def __init__(self, variable_name: str, exp: Node):
        super().__init__()
        self.variable = variable_name
        self.expression = exp

    def __repr__(self):
        return f'(L{self.variable}.{self.expression.__repr__()})'

class Application(Node):
    def __init__(self, term_a: Node, term_b: Node):
        super().__init__()
        self.fn = term_a
        self.operand = term_b

    def __repr__(self):
        return f'({self.fn.__repr__()}) ({self.operand.__repr__()})'

#
def parse_expression(token_list, do_app, depth):
    result = None
    prefix = '\t' * depth
    print_pre = lambda v: print(f'{prefix}{v}')
    print_pre('--------------------------------------------------')
    print_pre(f"parse_expression({token_list}, {do_app}, {depth})")

    # if we see Lambda, make sure we then have Variable Period
    if isinstance(token_list[0], Lambda):
        # if not Variable, error
        if not isinstance(token_list[1], VarToken):
            raise Exception('Variable must follow Lambda')
        # if not Variable Period, error
        elif not isinstance(token_list[2], Period):
            raise Exception('Period must follow {Lambda}{Variable}')

        print_pre(f"{depth}. Found Lambda: {token_list[:3]}, parsing: {token_list[3:]}\n\n")
        # sub_expression is chunked expression, then rest is rest of token list
        sub_expr, rest = parse_expression(token_list[3:], True, depth + 1)

        print_pre(f"\n-> Asbtraction: L{token_list[1].val}. {sub_expr}\n")
        # turn verified Lambda abstraction into Abstraction
        expr = Abstraction(token_list[1].val, sub_expr)
        print_pre(f"-> Parsing rest: {rest}")
        result = expr, rest

    # if we get just a raw variable, that's the term, return it and the rest separately
    elif isinstance(token_list[0], VarToken):
        # return variable and rest
        print_pre(f"{depth}. Found variable: {token_list[0].val}, parsing rest: {token_list[1:]}")
        result = Variable(token_list[0].val), token_list[1:]

    # if we see left paren
    elif isinstance(token_list[0], LParen):
        print_pre(f"{depth}. Found '(', parsing rest: {token_list[1:]}\n\n")
        sub_expr, rest = parse_expression(token_list[1:], True, depth + 1)
        print_pre(f"-> Sub expr: {sub_expr}, rest: {rest}")
        if not isinstance(rest[0], RParen):
            raise Exception('Unmatched left parenthesis')
        print_pre(f"-> paren-term: {sub_expr}")
        result = sub_expr, rest[1:]
    else:
        raise Exception("Invalid program")

    # after getting a valid expression built, unpack
    expr, rest = result
    # if a whitespace follows the expression
    if do_app:
        while rest != [] and isinstance(rest[0], Whitespace):
            print_pre(f"-> APPLYING because we have term on left and whitespace following:")
            # term_b of the application is FIRST expression of rest
            # do this simply by parsing rest
            print_pre(f"-> LEFT: {expr}, parsing: {rest}")
            right, rest = parse_expression(rest[1:], False, depth + 1)

            print_pre(f"-> RIGHT: {right}")
            # build Application
            print_pre(f"\n-> APPLICATION: {expr} <-> {right}\n")
            expr = Application(expr, right)

    print_pre('--------------------------------------------------')

    return expr, rest


def eval_expr(expr):

    if isinstance(expr, Abstraction):
        return Abstraction(expr.variable, eval_expr(expr.expression))

    if not isinstance(expr, Application):
        return expr

    if not isinstance(expr.fn, Abstraction):
        reduced_left = eval_expr(expr.fn)
        if not isinstance(reduced_left, Abstraction):
            return Application(reduced_left, expr.operand)

        return eval_expr(Application(reduced_left, expr.operand))


    # when applying (Lx. T1) T2
    # we just return T1[x->T2]
    return eval_expr(substitute_expr(expr.fn.variable, expr.operand, expr.fn.expression))

# (Lx.x y) T

# subst(x, 'x y', T)
# Application(subst(x, 'x', T), subst(x, 'y', T))

def substitute_expr(var_name: str, applicand, expr):
    if isinstance(expr, Variable):
        if expr.name == var_name:
            return applicand
        else:
            return expr
    elif isinstance(expr, Abstraction):
        if expr.variable == var_name:
            return expr
        else:
            inner_subbed_expr = substitute_expr(var_name, applicand, expr.expression)
            return Abstraction(expr.variable, inner_subbed_expr)
    elif isinstance(expr, Application):
        # What is (T1 T2)[var_name->applicand]
        # T1[var_name->applicand] T2[var_name->applicand]

        # variable and replacor is same, we're just distributing across both terms
        clean_sub = func.partial(substitute_expr, var_name, applicand)
        return Application(clean_sub(expr.fn), clean_sub(expr.operand))


def make_unique(expr, next_id):
    if isinstance(expr, Application):
        l, nxt = make_unique(expr.fn, next_id)
        r, nxt = make_unique(expr.operand, nxt)
        return Application(l, r), nxt
    elif isinstance(expr, Abstraction):
        replaced = substitute_expr(expr.variable, Variable(next_id), expr.expression)
        inner, nxt = make_unique(replaced, next_id + 1)
        return Abstraction(next_id, inner), nxt
    elif isinstance(expr, Variable):
        return expr, next_id

if __name__ == '__main__':
    # expression = '(Lx.a x) d'
    # expression = '(Lm.Ln.m (La.Lb.Lc.b (a b c)) n) (Lx.Ly.x x y) (Lx.Ly.x y)'
    expression = '(Lx.Ly.x x y) (Lx.Ly.x y)'


    print('--------START TOKENIZING---------')
    tokens = tokenize(expression)
    print('--------DONE TOKENIZING---------')
    print(f'TOKENS: {tokens}')

    print('--------START PARSING--------')
    parsed, _ = parse_expression(tokens, True, depth=0)
    print('--------DONE PARSING--------')
    print(f'PARSED EXPR: {parsed}')

    print('--------START EVALUATING--------')
    result = eval_expr(parsed)

    print('--------DONE EVALUATING--------')
    print(f'RESULT: {result}')
    # print(parse_expression(tokenize('a b c'), True, depth=0))
