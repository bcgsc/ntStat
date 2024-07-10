import tabulate


class TablePrinter:

    @staticmethod
    def get_valid_formats():
        return tabulate.tabulate_formats

    def __init__(self, fmt) -> None:
        self.__fmt = fmt

    def print(
        self,
        title: str,
        *rows: list[list[str]],
        header: list[str] = (),
    ) -> None:
        if title:
            print(f"{title}:")
        print(tabulate.tabulate(rows, header, self.__fmt, ".10f", ","))
        print()
