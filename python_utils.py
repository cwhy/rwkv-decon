def format_num(num, unit):
    num_str = "{:.1f}".format(num).replace(".", "_")
    return num_str.rstrip("0").rstrip("_") + unit


def num_short_form(num):
    if num == 0:
        return "0"
    abs_num = abs(num)
    sign = "-" if num < 0 else ""

    if abs_num < 1000:
        return str(num)
    elif abs_num < 1000000:
        return sign + format_num(abs_num / 1000, "K")
    elif abs_num < 1000000000:
        return sign + format_num(abs_num / 1000000, "M")
    else:
        return sign + format_num(abs_num / 1000000000, "B")


if __name__ == "__main__":
    def test_convert_num_to_str():
        assert num_short_form(0) == "0", "Error: expected '0', but got '{}'".format(num_short_form(0))
        assert num_short_form(1) == "1", "Error: expected '1', but got '{}'".format(num_short_form(1))
        assert num_short_form(10) == "10", "Error: expected '10', but got '{}'".format(num_short_form(10))
        assert num_short_form(100) == "100", "Error: expected '100', but got '{}'".format(num_short_form(100))
        assert num_short_form(999) == "999", "Error: expected '999', but got '{}'".format(num_short_form(999))
        assert num_short_form(1000) == "1K", "Error: expected '1K', but got '{}'".format(num_short_form(1000))
        assert num_short_form(1500) == "1_5K", "Error: expected '1_5K', but got '{}'".format(
            num_short_form(1500))
        assert num_short_form(1999) == "2K", "Error: expected '2K', but got '{}'".format(num_short_form(1999))
        assert num_short_form(1000000) == "1M", "Error: expected '1M', but got '{}'".format(
            num_short_form(1000000))
        assert num_short_form(1500000) == "1_5M", "Error: expected '1_5M', but got '{}'".format(
            num_short_form(1500000))
        assert num_short_form(999999999) == "1000M", "Error: expected '1000M', but got '{}'".format(
            num_short_form(999999999))
        assert num_short_form(1000000000) == "1B", "Error: expected '1B', but got '{}'".format(
            num_short_form(1000000000))
        assert num_short_form(1500000000) == "1_5B", "Error: expected '1_5B', but got '{}'".format(
            num_short_form(1500000000))
        assert num_short_form(-1000) == "-1K", "Error: expected '-1K', but got '{}'".format(
            num_short_form(-1000))
        assert num_short_form(-1500) == "-1_5K", "Error: expected '-1_5K', but got '{}'".format(
            num_short_form(-1500))
        assert num_short_form(-1999) == "-2K", "Error: expected '-2K', but got '{}'".format(
            num_short_form(-1999))
        assert num_short_form(-1000000) == "-1M", "Error: expected '-1M', but got '{}'".format(
            num_short_form(-1000000))
        assert num_short_form(-1500000) == "-1_5M", "Error: expected '-1_5M', but got '{}'".format(
            num_short_form(-1500000))
        assert num_short_form(-999999999) == "-1000M", "Error: expected '-1000M', but got '{}'".format(
            num_short_form(-999999999))
        assert num_short_form(-1000000000) == "-1B", "Error: expected '-1B', but got '{}'".format(
            num_short_form(-1000000000))
        assert num_short_form(-1500000000) == "-1_5B", "Error: expected '-1_5B', but got '{}'".format(
            num_short_form(-1500000000))
        print("All tests passed")


    test_convert_num_to_str()
