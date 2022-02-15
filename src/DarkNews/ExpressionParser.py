# This module is a customisation of fourFn.py from PyParsing module examples
# https://github.com/pyparsing/pyparsing/blob/203fa36d7ae6b79344e4bf13531b77c09f313793/examples/fourFn.py

# fourFn.py
#
# Demonstration of the pyparsing module, implementing a simple 4-function expression parser,
# with support for scientific notation, and symbols for e and pi.
# Extended to add exponentiation and simple built-in functions.
# Extended test cases, simplified pushFirst method.
# Removed unnecessary expr.suppress() call (thanks Nathaniel Peterson!), and added Group
# Changed fnumber to use a Regex, which is now the preferred method
# Reformatted to latest pypyparsing features, support multiple and variable args to functions
#
# Copyright 2003-2019 by Paul McGuire
#
from pyparsing import (
    Literal,
    Word,
    Group,
    Forward,
    alphas,
    alphanums,
    Regex,
    ParseException,
    CaselessKeyword,
    Suppress,
    delimitedList,
)
import math
import operator

class ExpressionParser:
    '''
        Parse an expression of the kind
        var_z = 5 + exp(4*sin(PI/3)) + var_a
        where it does both computation of the expression without calling eval,
        and substitution with other variables (var_a), previously defined.
        It stores the variables in a dictionary, which is passed by argument.
        It is the same dictionary on which it does the lookup.
    '''

    # map operator symbols to corresponding arithmetic operations
    epsilon = 1e-100
    opn = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "^": operator.pow,
    }

    # functions
    fn = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "exp": math.exp,
        "abs": abs,
        "trunc": int,
        "round": round,
        "sgn": lambda a: -1 if a < -ExpressionParser.epsilon else 1 if a > ExpressionParser.epsilon else 0,
    }

    class ParsingError(Exception):
        ''' Deal with errors coming only from this parser '''
        pass

    def __init__(self, parameters):
        self.parameters = parameters
        self._bnf = None
        self._current_stack = []
        # load the grammar
        self._BNF()

    def _BNF(self):
        '''
            It means Backus-Naur Form: this is the main definition of the grammar
            expop    :: '^'
            multop   :: '*' | '/'
            addop    :: '+' | '-'
            equalop  :: '='
            integer  :: ['+' | '-'] '0'..'9'+
            ident    :: [ a-zA-Z_ ][ a-zA-Z0-9_ ]*
            atom     :: PI | E | real | fn '(' expr ')' | '(' expr ')'
            factor   :: atom [ expop factor ]*
            term     :: factor [ multop factor ]*
            expr     :: variable equalop term [ addop term ]*
        '''
        if not self._bnf:
            # use CaselessKeyword for e and pi, to avoid accidentally matching
            # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
            # and CaselessKeyword only match whole words
            e = CaselessKeyword("E")
            pi = CaselessKeyword("PI")
            # fnumber = Combine(Word("+-"+nums, nums) +
            #                    Optional("." + Optional(Word(nums))) +
            #                    Optional(e + Word("+-"+nums, nums)))
            # or use provided pyparsing_common.number, but convert back to str:
            # fnumber = ppc.number().addParseAction(lambda t: str(t[0]))
            fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
            ident = Word(alphas, alphanums + "_$") # this is the name of a function

            plus, minus, mult, div = map(Literal, "+-*/")
            lpar, rpar = map(Suppress, "()")
            addop = plus | minus
            multop = mult | div
            expop = Literal("^")
            equalop = Literal("=")

            expr = Forward()
            expr_list = delimitedList(Group(expr))
            # add parse action that replaces the function identifier with a (name, number of args) tuple
            def insert_fn_argcount_tuple(t):
                fn = t.pop(0)
                num_args = len(t[0])
                t.insert(0, (fn, num_args))

            fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
                insert_fn_argcount_tuple
            )
            atom = (
                addop[...]
                + (
                    (fn_call | pi | e | fnumber | ident).setParseAction(self._push_first)
                    | Group(lpar + expr + rpar)
                )
            ).setParseAction(self._push_unary_minus)

            # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
            # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
            factor = Forward()
            factor <<= atom + (expop + factor).setParseAction(self._push_first)[...]
            term = factor + (multop + factor).setParseAction(self._push_first)[...]
            expr <<= term + (addop + term).setParseAction(self._push_first)[...] # differentiate from assignment in order to describe functions argument list
            assignment_expr = Forward()
            assignment_expr <<= (ident + equalop + expr).setParseAction(self._push_first) # _push_first should be given only to the full match otherwise if given when it matches singularly then it would repeat everything every time it backtraces
            self._bnf = assignment_expr
        return self._bnf

    def _push_first(self, toks):
        self._current_stack.append(toks[0])
        # print(self._current_stack, toks)

    def _push_value(self, toks):
        try:
            value = self.parameters[toks[0]]
        except KeyError:
            raise self.ParsingError(f"Variable '{toks[0]}' not defined.")
        self._current_stack.append(value)

    def _push_unary_minus(self, toks):
        for t in toks:
            if t == "-":
                self._current_stack.append("unary -")
            else:
                break
    
    def _evaluate_variable(self, s):
        op, num_args = s.pop(), 0
        if isinstance(op, tuple):
            op, num_args = op
        if op == "unary -":
            return -self._evaluate_variable(s)
        if op in "+-*/^":
            # note: operands are pushed onto the stack in reverse order
            op2 = self._evaluate_variable(s)
            op1 = self._evaluate_variable(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op in self.fn.keys():
            # note: args are pushed onto the stack in reverse order
            args = reversed([self._evaluate_variable(s) for _ in range(num_args)])
            return self.fn[op](*args)
        elif op[0].isalpha():
            try:
                return self.parameters[op]
            except KeyError:
                raise self.ParsingError(f"Invalid identifier '{op}'")
        else:
            # try to evaluate as int first, then as float if int fails
            try:
                return int(op)
            except ValueError:
                return float(op)

    def evaluate_stack(self, copy=False):
        # if copy == True, then don't consume the stack
        # get the first element of the stack, which must be the variable name
        if copy:
            stack = self._current_stack[:]
        var_name = stack.pop()
        # consistency check: should not be equal to E, e, PI, pi, Pi, or any other function name
        if (var_name.lower() in ['e', 'pi']) or (var_name in self.fn.keys()):
            raise self.ParsingError(f"Variable name '{var_name}' not allowed.")
        # evaluate the rest of the stack and assign the result to the variable
        # print(var_name, stack)
        self.parameters[var_name] = self._evaluate_variable(stack)
        return var_name, self.parameters[var_name]

    def parse_string(self, *args, **kwargs):
        # clean the stack
        self._current_stack[:] = []
        return self._bnf.parseString(*args, **kwargs)


if __name__ == "__main__":
    parser = ExpressionParser({})

    def test(s, expected_name, expected_value):
        try:
            results = parser.parse_string(s, parseAll=True)
            name, val = parser.evaluate_stack(copy=True)
        except ParseException as pe:
            print(s, "failed parse:", str(pe))
        except ExpressionParser.ParsingError as e:
            print(s, "failed eval:", str(e), parser._current_stack)
        else:
            if (name == expected_name) and (val == expected_value):
                print(s, "<=>", name, "=", val, results, "=>", parser._current_stack)
            else:
                print(s, "<=>", name, "=", val, '!!!', expected_name, "=", expected_value, results, "=>", parser._current_stack)
    
    class TEST:
        gD = 9.7543
        A = gD * 9.7543
        B = 9.7543**2 + gD
        C = 9.7543 + 2 - 4*A
        D = 9.7543 + math.sin(0.3)
        E = D
        alphaD = gD ** 2 / (4 * math.pi)
        sinx = math.sin(math.pi / 3) - 8

    test("gD = 9.7543", "gD", TEST.gD)
    test("A = gD * 9.7543", "A", TEST.A)
    test("B = 9.7543^2 + gD", "B", TEST.B)
    test("C = 9.7543 + 2 - 4*A", "C", TEST.C)
    test("D = 9.7543 + sin(0.3)", "D", TEST.D)
    test("E = 9.7543 + sin(0.3)", "E", TEST.E)
    test("alphaD = gD^2 / (4 * PI)", "alphaD", TEST.alphaD)
    test("sinx = sin(PI/3) - 8.", "sinx", TEST.sinx)
    test("number = sinx * 5.35e6", "number", TEST.sinx * 5.35e6)
    test("number2 = gD^2 * 5.35e6", "number", TEST.gD * 5.35e6)
    test("exp = exp(3*PI)", "exp", math.exp(3 * math.pi))
    test("exp_0 = exp(3*PI)", "exp_0", math.exp(3 * math.pi))
    test("24ff = -10+tan(PI/4)^2", "24ff", -10 + math.tan(math.pi / 4) ** 2)
    test("ff26 = -10+tan(PI/4)^2", "ff24", -10 + math.tan(math.pi / 4) ** 2)
    test(" ", "", 0)

    # print stored variables
    for k, v in parser.parameters.items():
        print(k, "=", v)


    # test("9", 9)
    # test("-9", -9)
    # test("--9", 9)
    # test("-E", -math.e)
    # test("9 + 3 + 6", 9 + 3 + 6)
    # test("9 + 3 / 11", 9 + 3.0 / 11)
    # test("(9 + 3)", (9 + 3))
    # test("(9+3) / 11", (9 + 3.0) / 11)
    # test("9 - 12 - 6", 9 - 12 - 6)
    # test("9 - (12 - 6)", 9 - (12 - 6))
    # test("2*3.14159", 2 * 3.14159)
    # test("3.1415926535*3.1415926535 / 10", 3.1415926535 * 3.1415926535 / 10)
    # test("PI * PI / 10", math.pi * math.pi / 10)
    # test("PI*PI/10", math.pi * math.pi / 10)
    # test("PI^2", math.pi ** 2)
    # test("round(PI^2)", round(math.pi ** 2))
    # test("6.02E23 * 8.048", 6.02e23 * 8.048)
    # test("e / 3", math.e / 3)
    # test("sin(PI/2)", math.sin(math.pi / 2))
    # test("10+sin(PI/4)^2", 10 + math.sin(math.pi / 4) ** 2)
    # test("trunc(E)", int(math.e))
    # test("trunc(-E)", int(-math.e))
    # test("round(E)", round(math.e))
    # test("round(-E)", round(-math.e))
    # test("E^PI", math.e ** math.pi)
    # test("exp(0)", 1)
    # test("exp(1)", math.e)
    # test("2^3^2", 2 ** 3 ** 2)
    # test("(2^3)^2", (2 ** 3) ** 2)
    # test("2^3+2", 2 ** 3 + 2)
    # test("2^3+5", 2 ** 3 + 5)
    # test("2^9", 2 ** 9)
    # test("sgn(-2)", -1)
    # test("sgn(0)", 0)
    # test("sgn(0.1)", 1)
    # test("foo(0.1)", None)
    # test("round(E, 3)", round(math.e, 3))
    # test("round(PI^2, 3)", round(math.pi ** 2, 3))
    # test("sgn(cos(PI/4))", 1)
    # test("sgn(cos(PI/2))", 0)
    # test("sgn(cos(PI*3/4))", -1)
    # test("+(sgn(cos(PI/4)))", 1)
    # test("-(sgn(cos(PI/4)))", -1)
    # test("hypot(3, 4)", 5)
    # test("multiply(3, 7)", 21)
    # test("all(1,1,1)", True)
    # test("all(1,1,1,1,1,0)", False)