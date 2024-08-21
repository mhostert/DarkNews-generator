#!/usr/bin/env python3

from pyparsing import ParseException
import math

from DarkNews import AssignmentParser
from DarkNews import GenLauncher
from DarkNews.ModelContainer import ModelContainer

from .helpers import assert_all, soft_assert


def test_ModelContainer_default():
    mc = vars(ModelContainer(loglevel="error"))
    gl = vars(GenLauncher(loglevel="error"))

    with assert_all() as assertions:
        for key, val in mc.items():
            if key in gl.keys():
                assertions.append(
                    soft_assert(
                        mc[key] == gl[key] or isinstance(mc[key], list),
                        f"Different values between GenLauncher and ModelContainer: {key}: ModelContainer={mc[key]}, GenLauncher={gl[key]}",
                    )
                )


def test_input_parameter_files():
    gen1 = GenLauncher(param_file="tests/test_parameter_file_3portal.txt", loglevel="ERROR")
    gen2 = GenLauncher(param_file="tests/test_parameter_file_generic.txt", loglevel="ERROR")

    assert gen1.bsm_model.epsilon == 1e-4
    assert gen2.bsm_model.ceV == 1e-4


def test_argparse():
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
                return True
            else:
                print("[\033[91mNO\033[0m]", s, "<=>", name, "=", val, "!!!", expected_name, "=", expected_value, results, "=>", parser._current_stack)
                return False

    class TEST:
        gD = 9.7543
        A = gD * 9.7543
        B = 9.7543**2 + gD
        C = 9.7543 + 2 - 4 * A
        D = 9.7543 + math.sin(0.3)
        E = D
        alphaD = gD**2 / (4 * math.pi)
        sinx = math.sin(math.pi / 3) - 8
        hbar = 6.582119569e-25
        c = 299792458.0
        a_variable = c**2 * 3.2e-4 / math.sin(math.pi / 7) + 12 * math.exp(-2 * abs(hbar))

    test("gD = 9.7543", "gD", TEST.gD)
    test("A = gD * 9.7543", "A", TEST.A)
    test("B = 9.7543^2 + gD", "B", TEST.B)
    test("C = 9.7543 + 2 - 4*A", "C", TEST.C)
    test("D = 9.7543 + sin(0.3)", "D", TEST.D)
    test("E = 9.7543 + sin(0.3)", "E", TEST.E, should_fail=True)
    test("alphaD = gD^2 / (4 * PI)", "alphaD", TEST.alphaD)
    test("sinx = sin(PI/3) - 8.", "sinx", TEST.sinx)
    test("number = sinx * 5.35e6", "number", TEST.sinx * 5.35e6)
    test("number2 = gD^2 * 5.35e6", "number2", TEST.gD**2 * 5.35e6)
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
    # test('list_5 = ["hello world" "people", 10]', "list_5", None, should_fail=True)
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


if __name__ == "__main__":
    test_argparse()
