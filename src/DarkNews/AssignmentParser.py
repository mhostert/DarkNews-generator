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
from xmlrpc.client import Boolean
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
    Or,
    QuotedString,
)
import math
import operator


class AssignmentParser:
    """
        Parse an expression of the kind
        var_z = 5 + exp(4*sin(PI/3)) + var_a
        where it does both computation of the expression without calling eval,
        and substitution with other variables (var_a), previously defined.
        It stores the variables in a dictionary, which is passed by argument.
        It is the same dictionary on which it does the lookup.
    """

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
        "sgn": lambda a: -1 if a < -AssignmentParser.epsilon else 1 if a > AssignmentParser.epsilon else 0,
        "list": lambda *args: list([*args]),
        "sum": sum,
        "str": str,
    }

    class ParsingError(Exception):
        """ Deal with errors coming only from this parser """

        pass

    def __init__(self, parameters):
        self.parameters = parameters
        self._bnf = None
        self._current_stack = []
        # load the grammar
        self._BNF()

    def _BNF(self):
        """
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
        """
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
            ident = Word(alphas, alphanums + "_")  # variables can have names starting only with letters, no underscore nor numbers

            plus, minus, mult, div = map(Literal, "+-*/")
            lpar, rpar = map(Suppress, "()")
            l_delim, r_delim = map(Suppress, "[]")
            addop = plus | minus
            multop = mult | div
            expop = Literal("^")
            equalop = Literal("=")

            expr = Forward()
            expr_list = delimitedList(Group(expr))

            # Quoted strings
            quoted_str_class = QuotedString(quoteChar='"', escChar="\\", multiline=True, unquoteResults=False) | QuotedString(  # double quote string
                quoteChar="'", escChar="\\", multiline=True, unquoteResults=False
            )  # single quote string  # keep the quotes and use them to separate strings from variables
            quoted_str = quoted_str_class.setParseAction(self._push_first)

            # Lists
            list_elements = delimitedList(Group(quoted_str | expr))
            list_pattern = l_delim - Group(list_elements) + r_delim

            # Function call: need to encapsulate again list_pattern in a group otherwise sum([list]) would detect one more argument, but it is only one: that's because the list is converted to [('list', n), [...n elements...]], so 2 elements, but it is only one if it is a parameter of a function call
            fn_call = (ident + lpar - Group(expr_list | Group(list_pattern)) + rpar).setParseAction(self._insert_fn_argcount_tuple)
            atom = (addop[...] + ((fn_call | pi | e | fnumber | ident).setParseAction(self._push_first) | Group(lpar + expr + rpar))).setParseAction(
                self._push_unary_minus
            )

            # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
            # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
            factor = Forward()
            factor <<= atom + (expop + factor).setParseAction(self._push_first)[...]
            term = factor + (multop + factor).setParseAction(self._push_first)[...]
            expr <<= term + (addop + term).setParseAction(self._push_first)[...]
            assignment_expr = Forward()
            # _push_first should be given only to the full match otherwise if given when it matches singularly then it would repeat everything every time it backtracks
            assignment_expr <<= (
                ident
                + equalop
                + Or(
                    [
                        quoted_str,
                        list_pattern.setParseAction(self._insert_list_argcount_tuple, self._push_first),
                        expr,  # this is the most-specific patterns should be placed ahead
                    ]
                )
            ).setParseAction(self._push_first)

            # assignment_expr <<= (ident + equalop + list_pattern.setParseAction(self._insert_list_argcount_tuple, self._push_first)).setParseAction(self._push_first)
            self._bnf = assignment_expr
        return self._bnf

    def _insert_fn_argcount_tuple(self, t):
        """ Parse action that replaces the function identifier with a (name, number of args, "function") tuple. """
        fn = t.pop(0)
        num_args = len(t[0])
        t.insert(0, (fn, num_args))

    def _insert_list_argcount_tuple(self, t):
        """ Parse action that replaces the list found with a ("list", number of elements) tuple. """
        num_args = len(t[0])
        t.insert(0, ("list", num_args))

    def _push_first(self, toks):
        self._current_stack.append(toks[0])

    def _push_unary_minus(self, toks):
        for t in toks:
            if t == "-":
                self._current_stack.append("unary -")
            else:
                break

    def _evaluate_expression(self, s):
        op, num_args = s.pop(), 0
        if isinstance(op, tuple):
            op, num_args = op
        if isinstance(op, bool):
            return op
        if op == "unary -":
            return -self._evaluate_expression(s)
        if op in "+-*/^":
            # note: operands are pushed onto the stack in reverse order
            op2 = self._evaluate_expression(s)
            op1 = self._evaluate_expression(s)
            return self.opn[op](op1, op2)
        elif op == "PI":
            return math.pi  # 3.1415926535
        elif op == "E":
            return math.e  # 2.718281828
        elif op.lower() == "true":
            return True
        elif op.lower() == "false":
            return False
        elif op == "str":  # so far str will consume only one argument
            return str(s.pop())
        elif op in self.fn.keys():
            # note: args are pushed onto the stack in reverse order
            args = reversed([self._evaluate_expression(s) for _ in range(num_args)])
            return self.fn[op](*args)
        elif op[0].isalpha():
            try:
                return self.parameters[op]
            except KeyError:
                raise self.ParsingError(f"Invalid identifier '{op}'")
        else:
            # try to evaluate as float, then as string
            try:
                return float(op)
            except ValueError:
                return str(op.strip('"').strip("'"))

    def evaluate_stack(self, copy=False):
        # if copy == True, then don't consume the stack
        # get the first element of the stack, which must be the variable name
        if copy:
            stack = self._current_stack[:]
        else:
            stack = self._current_stack
        var_name = stack.pop()
        # consistency check: should not be equal to E, e, PI, pi, Pi, or any other function name
        if (var_name.lower() in ["e", "pi", "true", "false"]) or (var_name in self.fn.keys()):
            raise self.ParsingError(f"Variable name '{var_name}' not allowed.")
        # evaluate the rest of the stack and assign the result to the variable
        self.parameters[var_name] = self._evaluate_expression(
            stack
        )  # if evaluation contains the new variable which we are trying to assign, everything will fail by construction
        return var_name, self.parameters[var_name]

    def clean_stack(self):
        self._current_stack[:] = []

    def parse_string(self, *args, **kwargs):
        self.clean_stack()
        return self._bnf.parseString(*args, **kwargs)

    def scan_string(self, *args, **kwargs):
        self.clean_stack()
        return self._bnf.scanString(*args, **kwargs)

    def parse_file(self, file, comments="#"):
        # read file
        try:
            # assume it is an open file
            lines = file.readlines()
        except AttributeError:
            # assume it is a str to file path
            with open(file, "r", encoding="utf8") as f:
                lines = f.readlines()

        # create a clean text without blank lines and comments
        clean_text = ""
        for i, line in enumerate(lines):
            partition = line.partition(comments)[0]
            if partition.strip() == "":
                continue
            clean_text += partition + "\n"
        for tokens, i_beg, i_end in self.scan_string(clean_text):
            # the effective parsing is done when the generator is called
            # so the stack fills at each iteration
            try:
                self.evaluate_stack()
            except ParseException as pe:
                raise self.ParseException(" ".join(tokens) + f" Failed parse (start: {i_beg}, end: {i_end}): " + str(pe))
            except self.ParsingError as e:
                raise self.ParsingError(" ".join(tokens) + f" Failed evaluation (start: {i_beg}, end: {i_end}): " + str(e))
            finally:
                self.clean_stack()  # clean the stack in any case, because if there are errors, then we need to have a clean list before the next iteration


if __name__ == "__main__":
    parser = AssignmentParser({})

    def test(s, expected_name, expected_value, should_fail=False):
        should_fail_string = "[\033[92mOK\033[0m]" if should_fail else "[\033[91mNO\033[0m]"
        try:
            results = parser.parse_string(s, parseAll=True)
            name, val = parser.evaluate_stack(copy=True)
        except ParseException as pe:
            print(should_fail_string, s, "failed parse:", str(pe))
        except AssignmentParser.ParsingError as e:
            print(should_fail_string, s, "failed eval:", str(e), parser._current_stack)
        else:
            condition_value = (val == expected_value) if not isinstance(expected_value, list) else all([v == ev for v, ev in zip(val, expected_value)])
            if (name == expected_name) and condition_value:
                print("[\033[92mOK\033[0m]", s, "<=>", name, "=", val, results, "=>", parser._current_stack)
            else:
                print("[\033[91mNO\033[0m]", s, "<=>", name, "=", val, "!!!", expected_name, "=", expected_value, results, "=>", parser._current_stack)

    class TEST:
        gD = 9.7543
        A = gD * 9.7543
        B = 9.7543 ** 2 + gD
        C = 9.7543 + 2 - 4 * A
        D = 9.7543 + math.sin(0.3)
        E = D
        alphaD = gD ** 2 / (4 * math.pi)
        sinx = math.sin(math.pi / 3) - 8
        hbar = 6.582119569e-25
        c = 299792458.0
        a_variable = c ** 2 * 3.2e-4 / math.sin(math.pi / 7) + 12 * math.exp(-2 * abs(hbar))

    test("gD = 9.7543", "gD", TEST.gD)
    test("A = gD * 9.7543", "A", TEST.A)
    test("B = 9.7543^2 + gD", "B", TEST.B)
    test("C = 9.7543 + 2 - 4*A", "C", TEST.C)
    test("D = 9.7543 + sin(0.3)", "D", TEST.D)
    test("E = 9.7543 + sin(0.3)", "E", TEST.E, should_fail=True)
    test("alphaD = gD^2 / (4 * PI)", "alphaD", TEST.alphaD)
    test("sinx = sin(PI/3) - 8.", "sinx", TEST.sinx)
    test("number = sinx * 5.35e6", "number", TEST.sinx * 5.35e6)
    test("number2 = gD^2 * 5.35e6", "number2", TEST.gD ** 2 * 5.35e6)
    test("exp = exp(3*PI)", "exp", math.exp(3 * math.pi), should_fail=True)
    test("exp_0 = exp(3*PI)", "exp_0", math.exp(3 * math.pi))
    test("24ff = -10+tan(PI/4)^2", "24ff", -10 + math.tan(math.pi / 4) ** 2, should_fail=True)
    test("ff24 = -10+tan(PI/4)^2", "ff24", -10 + math.tan(math.pi / 4) ** 2)
    test(" ", "", 0, should_fail=True)
    test("hbar = 6.582119569e-25", "hbar", TEST.hbar)
    test("c = 299792458.0", "c", TEST.c)
    test("a_variable = c^2 * 3.2e-4 / sin(PI/7) + 12 * exp( -2 * abs(hbar) )", "a_variable", TEST.a_variable)
    test(
        """multi_line = c^2 * 3.2e-4 / sin(PI/7) + 
    12 * exp( -2 * abs(hbar) )""",
        "multi_line",
        TEST.a_variable,
    )
    test('s_1 = "hello world"', "s_1", "hello world")
    test('s_2 = "hello world" "people"', "s_2", None, should_fail=True)
    test("s_3 = 'hello world'", "s_3", "hello world")
    test("test_1 = true", "test_1", True)
    test("test_2 = False", "test_2", False)
    test("test_3 = True False", "test_3", None, should_fail=True)
    test('test_4 = True "hello"', "test_4", None, should_fail=True)
    test('test_5 = exp(3*PI)*c True "hello"', "test_5", None, should_fail=True)
    test('test_6 = exp(3*PI)*c + True - "hello"', "test_6", None, should_fail=True)
    test("test_7 = exp(3*PI)*c + True", "test_7", math.exp(3 * math.pi) * TEST.c + True)  # it's ok, because float + bool is the bool converted to integer
    test("test_8 = False + exp(3*PI)*c", "test_8", False + math.exp(3 * math.pi) * TEST.c)
    test("list_1 = [-10, 2.3]", "list_1", [-10, 2.3])
    test("sum_1 = sum(list_1)", "sum_1", sum([-10, 2.3]))
    test("sum_2 = sum([-10+tan(PI/4)^2, exp(3*PI)*E])", "sum_2", sum([-10 + math.tan(math.pi / 4) ** 2, math.exp(3 * math.pi) * math.e]))
    test("list_2 = [-10+tan(PI/4)^2, exp(3*PI)*E]", "list_2", [-10 + math.tan(math.pi / 4) ** 2, math.exp(3 * math.pi) * math.e])
    test('list_3 = ["hello", exp(3*PI)*E]', "list_3", ["hello", math.exp(3 * math.pi) * math.e])
    test('list_4 = ["hello", "world"]', "list_4", ["hello", "world"])
    test('list_5 = ["hello world" "people", 10]', "list_5", None, should_fail=True)
    test('list_6 = ["hello", a_variable, "world"]', "list_6", ["hello", TEST.a_variable, "world"])
    test(
        """list_7 = [
                \"hello\",
                a_variable,
                \"world\",
                \"Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nunc ullamcorper blandit nibh, vitae congue lacus convallis at.
Donec interdum, ex et fermentum aliquam, ante magna convallis magna, et molestie quam quam et dolor.\",
                c^2 * 3.2e-4 / sin(PI/7) + 
    12 * exp( -2 * abs(hbar) )
            ]""",
        "list_7",
        [
            "hello",
            TEST.a_variable,
            "world",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\nNunc ullamcorper blandit nibh, vitae congue lacus convallis at.\nDonec interdum, ex et fermentum aliquam, ante magna convallis magna, et molestie quam quam et dolor.",
            TEST.a_variable,
        ],
    )
    test(
        """nuclear_targets = [
    'H1',
    'C12',
    'O16',
    'Cu63',
    "Zn64",
    'Pb208'
]""",
        "nuclear_targets",
        ["H1", "C12", "O16", "Cu63", "Zn64", "Pb208"],
    )
    test(
        """fiducial_mass_per_target = [
    0.42, 
    5.4, 
    3.32, 
    0.4, 
    0.8, 
    12.0
] """,
        "fiducial_mass_per_target",
        [0.42, 5.4, 3.32, 0.4, 0.8, 12.0],
    )
    test("fiducial_mass = sum(fiducial_mass_per_target)", "fiducial_mass", sum([0.42, 5.4, 3.32, 0.4, 0.8, 12.0]))

    # print stored variables
    print("\nStored variables")
    for k, v in parser.parameters.items():
        print(k, "=", v, type(v))

    # test multi assignments
    parser_multi = AssignmentParser({})
    text = """
hbar = 6.582119569e-25
c = 299792458.0

a_variable = -10+tan(PI/4)^2
a_list = [
                \"hello\",
                a_variable,
                \"world\",
                \"Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Nunc ullamcorper blandit nibh, vitae congue lacus convallis at.
Donec interdum, ex et fermentum aliquam, ante magna convallis magna, et molestie quam quam et dolor.\",
                c^2 * 3.2e-4 / sin(PI/7) + 
    12 * exp( -2 * abs(hbar) )
            ]


sinx = sin(PI/3) - 8.

a_multi_line_expression = c^2 * 3.2e-4 / sin(PI/7) + 
    12 * exp( -2 * abs(hbar) )
a_bool = True

a_string = \"hello world\"
    """

    matches = parser_multi.scan_string(text)
    for _ in matches:
        try:
            print(parser_multi._current_stack)
            name, val = parser_multi.evaluate_stack(copy=True)
        except ParseException as pe:
            print("failed parse:", str(pe))
        except AssignmentParser.ParsingError as e:
            print("failed eval:", str(e), parser_multi._current_stack)
        finally:
            parser_multi._current_stack[:] = []

    print("\nStored variables")
    for k, v in parser_multi.parameters.items():
        print(k, "=", v, type(v))
