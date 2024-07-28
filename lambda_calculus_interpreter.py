"""
- lexing (turning strings into named tokens we can work with)
- parsing (turn tokens into tree)
- interpret (take a tree and interpret directly)
"""
import re
import functools as func
import time

# global variable used to Alpha-reduce expressions with same variable name
highest = 0

def build_token_grabber():
    """
    Construct RegEx string for tokenizer
    :return: RegEx string for re.match based on all token types
    """
    token_dict = {
        't_lambda': r'L',
        't_left_paren': r'\(',
        't_right_paren': r'\)',
        't_variable': r'[a-z]+',
        't_dot': r'\.',
        't_whitespace': r'\s+'
    }
    dct = token_dict
    return r'^(' + '|'.join([ f'(?P<{k}>{dct[k]})' for k in dct ]) + ')'


class Token:
    """
    Base class for token types
    Essentially just for REPR to return name of which Token child instance it is
    """
    def __init__(self):
        pass

    def __repr__(self):
        return self.__class__.__name__


class LParen(Token):
    """
    Left parenthesis Token
    """
    def __init__(self):
        super().__init__()


class RParen(Token):
    """
    Right parenthesis Token
    """
    def __init__(self):
        super().__init__()


class Lambda(Token):
    """
    lambda call ("L") Token
    """
    def __init__(self):
        super().__init__()


class VarToken(Token):
    """
    Token for any variable
    - Attribute VAL is the value or name of the variable
    """
    def __init__(self, val: str):
        super().__init__()
        self.val = val
    def __repr__(self):
        """
        Give instance name, not class name
        :return: name/VAL of variable, not the name VarToken
        """
        return super().__repr__() + f': {self.val}'


class Whitespace(Token):
    """
    Whitespace token
    """
    def __init__(self):
        super().__init__()


class Period(Token):
    """
    Period Token
    """
    def __init__(self):
        super().__init__()


def take_next_token(text: str) -> tuple:
    """
    Called many times in tokenize()
    Gets and creates token class instance of next closest token (space separated)

    :param text: Lambda Calculus source code
    :return: Next token, rest of code
    """
    # return token type (in terms of our string dict) of NEXT token
    match_result = re.match(build_token_grabber(), text)
    token_result = None

    if match_result is None:
        # nothing in the remainder of the file matched any token
        raise Exception("Lambda invalid syntax")

    # get token and get rest
    span = match_result.span()
    token = text[span[0]:span[1]]
    rest = text[span[1]:]

    dct = match_result.groupdict()
    for k in dct:
        mtch = dct[k]
        if mtch is None:
            continue
        # catch which type of token the token we got is and invoke that Class
        if k == 't_lambda':
            token_result = Lambda()
        elif k == 't_left_paren':
            token_result = LParen()
        elif k == 't_right_paren':
            token_result = RParen()
        elif k == 't_variable':
            token_result = VarToken(token)
        elif k == 't_dot':
            token_result = Period()
        elif k == 't_whitespace':
            token_result = Whitespace()

    return token_result, rest


def tokenize(text: str):
    """
    Lexer of source code
    takes in raw text and spits out list of tokens

    :param text: Lambda Calculus source code
    :return: list of all tokens in the source code
    """
    tokens_list = []

    while len(text) > 0:
        tok, rest = take_next_token(text)
        tokens_list.append(tok)
        text = rest
    return tokens_list


class Node:
    """
    base class of three kinds of Terms
    """
    def __init__(self):
        pass


class Variable(Node):
    """
    Variable node (Ex. 'x')

    Attr NAME is the character/name/value of the variable
    """
    def __init__(self, variable_name: str):
        super().__init__()
        self.name = variable_name

    def __repr__(self):
        return self.name


class Abstraction(Node):
    """
    Abstraction term node (Ex. 'Lx.Ly.x')

    Attr VARIABLE is the name of the bound variable (Ex. would be 'x')
    Attr EXPRESSION is the inner expression of function (Ex. would be Abstraction 'Ly.x')
    """
    def __init__(self, variable_name: str, exp: Node):
        super().__init__()
        self.variable = variable_name
        self.expression = exp

    def __repr__(self):
        """
        Display the actual whole term (wrapped in parentheses)
        :return: "(Lx.Ly.x)"
        """
        return f'(L{self.variable}.{self.expression.__repr__()})'

    # def __str__(self):
    #     succ_var = self.variable
    #     x = 0
    #     for s in self.expression.__repr__():
    #         if s == succ_var:
    #             x += 1
    #     return str(x)


class Application(Node):
    """
    Application term node (Ex. "x y", x applied to y)

    Attr FN is left term (the function or value being applied to the right) (Ex. 'x')
    Attr OPERAND is the right term (Ex. 'y')
    """
    def __init__(self, term_a: Node, term_b: Node):
        super().__init__()
        self.fn = term_a
        self.operand = term_b

    def __repr__(self):
        """
        Display the actual application of two terms (wrapped in parentheses)
        :return: '(x y)'
        """
        return f'({self.fn.__repr__()}) ({self.operand.__repr__()})'


def parse_expression(token_list, do_app, depth):
    """
    Parses an entire expression of Lambda Calculus
    :param token_list: List of all tokens in source code
    :param do_app: On/Off flag telling is whether we want to apply terms in inner-expressions (yes for Ab and Paren)
    :param depth: Depth of recursive calls to parse (0 is top level)
    :return: Expression in terms of abstract classes, ready for evaluator and rest of source code (eventually [])
    """
    result = None
    # for printing for clarity
    prefix = '\t' * depth
    # special print to include prefixes. also can be turned off
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

        print_pre(f"Found Lambda: {token_list[:3]}, parsing: {token_list[3:]}\n\n")
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
        print_pre(f"Found variable: {token_list[0].val}, parsing rest: {token_list[1:]}")
        result = Variable(token_list[0].val), token_list[1:]

    # if we see left paren
    elif isinstance(token_list[0], LParen):
        print_pre(f"Found '(', parsing rest: {token_list[1:]}\n\n")
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


def eval_expr(expr, depth):
    """
    Reduces entire expression of Lambda Calculus
    :param expr: Expression given from parse_expression()[0]
    :return: Reduced Lambda Calculus expression
    """
    prefix = '\t' * depth
    # special print to include prefixes. also can be turned off
    print_pre = lambda v: print(f'{time.sleep(.0005)}{prefix}{v}')


    global highest
    if isinstance(expr, Abstraction):
        # if Abstraction, then recursively evaluate and reduce inner expression then return rewritten Abstraction
        print_pre(f"Abstraction: {expr}")
        return Abstraction(expr.variable, eval_expr(expr.expression, depth + 1))

    # if variable, leave
    # if isinstance(expr, Variable) ???
    if not isinstance(expr, Application):
        print_pre(f"Eval Variable, pass: {expr}")
        return expr

    # must be some kind of Application henceforth
    print_pre(f"Application: {expr}")
    if not isinstance(expr.fn, Abstraction):
        print_pre(f"Evaluating left operator: {expr.fn}")
        # if left is NOT an Abstraction, first try to reduce left term
        reduced_left = eval_expr(expr.fn, depth + 1)
        if not isinstance(reduced_left, Abstraction):
            # if still not an abstraction, then we apply LEFT term to recursively evaluated RIGHT term
            print_pre(f"Still not an abstraction ({reduced_left}) so applying to evaluated ({expr.operand})")
            return Application(reduced_left, eval_expr(expr.operand, depth + 1))

        # if left became an abstraction, then we just apply it to recursively evaluated RIGHT term
        print_pre(f"Evaluate: {reduced_left} <-> EVAL({expr.operand})")
        return eval_expr(Application(reduced_left, eval_expr(expr.operand, depth + 1)), depth + 1)

    # Alpha reduction, making sure no redundant variable names
    # get names of variables in LEFT of Application
    # to_replace = get_variable_names(eval_expr(expr.fn, depth + 1))
    # # RIGHT term
    # op = expr.operand
    # loop through RIGHT and replace any problematic variables with numbers to ensure no further repeats
    # for v in to_replace:
    #     op = rename_variable(v, v[0]+str(highest), op)
    #     highest += 1

    # substitute RIGHT in to LEFT (replace LEFT variable, with RIGHT expression, in LEFT expression)
    # then reduce


    # return eval_expr(substitute_expr(expr.fn.variable, op, expr.fn.expression), depth + 1)
    return eval_expr(substitute_expr(expr.fn.expression, expr.fn.variable, expr.operand), depth + 1)


def get_variable_names(expr):
    """
    Get names of all variables (yes, even bound to ENSURE no issues) in LEFT function of Application
    :param expr: LEFT expression
    :return: set of variable names
    """
    if isinstance(expr, Variable):
        return { expr.name }
    elif isinstance(expr, Application):
        return get_variable_names(expr.fn) | get_variable_names(expr.operand)
    elif isinstance(expr, Abstraction):
        return {expr.variable} | get_variable_names(expr.expression)

def rename_variable(old : str, new : str, expr):
    """
    Rename variables in RIGHT term of Application with novel numbers to ensure no overloaded variables
    :param old: Old name of variable
    :param new: New name given by global HIGHEST
    :param expr: Expression in which replacements are done
    :return:
    """
    if isinstance(expr, Variable):
        if new == expr.name:
            print(f"Renaming: {expr} -> {new}")
        return Variable(new if expr.name == old else expr.name)
    elif isinstance(expr, Application):
        return Application(rename_variable(old, new, expr.fn), rename_variable(old, new, expr.operand))
    elif isinstance(expr, Abstraction):
        if expr.variable == old:
            print(f"quasi renaming {old} -> {new}")
            return Abstraction(new, rename_variable(old, new, expr.expression))
        print(f"renaming {old} -> {new}")
        return Abstraction(expr.variable, rename_variable(old, new, expr.expression))

def is_free(variable, expr):
    """
    :param variable: variable in question that were searching for in expression
    :param expr: expression in which we're searching
    :return: True if variable is free in expr, false if not
    """
    if isinstance(expr, Variable):
        return variable == expr.name

    elif isinstance(expr, Abstraction):
        if variable == expr.variable:
            return False
        return is_free(variable, expr.expression)

    elif isinstance(expr, Application):
        return is_free(variable, expr.fn) or is_free(variable, expr.operand)


def substitute_expr(var_name: str, applicand, expr):
    global highest
    if isinstance(expr, Variable):
        if expr.name == var_name:
            return applicand
        else:
            return expr
    elif isinstance(expr, Abstraction):
        # Ly.Lx.
        # if x            =     y
        if expr.variable == var_name:
            return expr
        else:
            # find variables in expr.expression
            abs_body = expr.expression
            abs_var = expr.variable
            if is_free(applicand, abs_var):
                abs_var = str(highest)
                abs_body = substitute_expr(expr.variable, Variable(abs_var), abs_body)
                highest += 1
                # return Abstraction(new_var, substitute_expr(var_name, applicand, new_body))
            inner_subbed_expr = substitute_expr(var_name, applicand, abs_body)
            return Abstraction(abs_var, inner_subbed_expr)
    elif isinstance(expr, Application):
        # What is (T1 T2)[var_name->applicand]
        # T1[var_name->applicand] T2[var_name->applicand]

        # variable and replacor is same, we're just distributing across both terms
        # clean_sub = func.partial(substitute_expr, var_name, applicand)
        # subbed_left = clean_sub(expr.fn)
        # subbed_right = clean_sub(expr.operand)
        # return Application(subbed_left, subbed_right)
        return Application(substitute_expr(var_name, applicand, expr.fn), substitute_expr(var_name, applicand, expr.operand))


# def make_unique(expr, next_id):
#     if isinstance(expr, Application):
#         l, nxt = make_unique(expr.fn, next_id)
#         r, nxt = make_unique(expr.operand, nxt)
#         return Application(l, r), nxt
#     elif isinstance(expr, Abstraction):
#         replaced = substitute_expr(expr.variable, Variable(next_id), expr.expression)
#         inner, nxt = make_unique(replaced, next_id + 1)
#         return Abstraction(next_id, inner), nxt
#     elif isinstance(expr, Variable):
#         return expr, next_id

if __name__ == '__main__':
    def lambda_interpret(lambda_str):

        print('--------START TOKENIZING---------')
        tokens = tokenize(lambda_str)
        print('--------DONE TOKENIZING---------')
        print(f'TOKENS: {tokens}')

        print('--------START PARSING--------')
        parsed, _ = parse_expression(tokens, True, depth=0)
        print('--------DONE PARSING--------')
        print(f'PARSED EXPR: {parsed}')

        print('--------START EVALUATING--------')
        result = eval_expr(parsed, depth=0)
        print('--------DONE EVALUATING--------')


        print(f'\n\n\nRESULT: {result}')
        return result

    def church_num_from_int(n: int):
        if n == 0:
            return f"(Ls.Lz.z)"
        elif n == 1:
            return f"(Ls.Lz.s z)"
        return f"(Ls.Lz.s {'(s '*(n-1) } z{')'*(n-1)})"

    test_strings = {
        'simple_sub': '(Lx.a x) d',
        'ADD 2 1': '(Lm.Ln.m (La.Lb.Lc.b (a b c)) n) (Lx.Ly.x (x y)) (Lx.Ly.x y)',
        'SCC 0': '(Ln.Ls.Lz.s (n s z)) (Ls.Lz.z)',
        'SCC 1': '(Ln.Ls.Lz.s (n s z)) (Ls.Lz.s z)',
        # this is technically correct, however the variables are rather ambiguous
        '3 2 (2^3)': '(Lx.Ly.x (x (x y))) (La.Lb.a (a b))',
        #
        'TRU x y': '(Lx.Ly.x) x y',
        'FLS x y': '(Lx.Ly.y) x y',
        '6 2': '(Lx.Ly.x (x (x (x (x (x y)))))) (La.Lb.a (a b))'

    }

    scc = '(Ln.Ls.Lz.s (n s z))'
    # add = f'(Lm.Ln.m ({scc} n))'
    add = f'(Lx.Ly.x {scc} y)'
    # mul = f'(Lm.Ln.m ((Lf.Lg.f ((Lh.Lj.Lk.j (h j k)) g)) n) (Ls.Lz.z))'
    mul = f"(Lm.Ln.m ({add} n) (Ls.Lz.z))"
    church = church_num_from_int
    # res = lambda_interpret(f'{add} ({church_num} {church_num})')
    # res = lambda_interpret('(Ln.Ls.Lz.s (n s z)) ((Ln.Ls.Lz.s (n s z)) (Lq.Lr.q r))')

    # mul = f"(Lm.Ln.m ({add} n) (Ls.Lz.z))"
    # this is MUL 4 3
    # this works

    # add and scc also works fully fine. hand typing MUL with 4 and 3 plugged in works, but when
    #           using our desired abstraction for MUL, it doesnt
    # res = lambda_interpret(f"{church(4)} ({add} {church(3)}) ({church(0)})")

    # res = lambda_interpret(f"{mul} {church(4)} {church(3)}")
    lambda_interpret(f"(((Lm.(Ln.((m) (((Lx.(Ly.((x) ((Ln.(Ls.(Lz.(s) (((n) (s)) (z))))))) (y)))) (n))) ((Ls.(Lz.z)))))) ((Ls.(Lz.(s) ((s) ((s) ((s) (z)))))))) ((Ls.(Lz.(s) ((s) ((s) (z))))))")
    """
    Notes
    
    - need to improve Alpha reduction, because while technically correct, sometimes answers make no sense
    ---see 3 2 (2^3)
    
    - does this work for a whole file?
    ---right now our reducer only works for one expression. 
    
    
    
    Ok right now, it can only really handle SCC, ADD is hard
    """
