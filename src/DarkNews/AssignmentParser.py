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
        elif op.lower() == "none":
            return None
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
