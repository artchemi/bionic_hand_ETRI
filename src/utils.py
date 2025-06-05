import numpy as numpy


def split_into_batches(data, window_size: int, step: int) -> list:
    """Разделяет всю серию ЭМГ или меток по окнам

    Args:
        data (_type_): ЭМГ или метки
        window_size (int): размер окна

    Returns:
        list: Список батчей
    """
    # return [data[i:i + window_size] for i in range(0, len(data), window_size)]
    return [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]
