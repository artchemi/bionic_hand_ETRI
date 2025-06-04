import numpy as numpy


def split_into_batches(data, batch_size: int) -> list:
    """Разделяет всю серию ЭМГ или меток по батчам

    Args:
        data (_type_): ЭМГ или метки
        batch_size (int): размер окна

    Returns:
        list: Список батчей
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
