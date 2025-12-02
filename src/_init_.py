"""
src/utils.py — Utilidades comunes.
"""
from pandas import Timestamp
from pandas.tseries.offsets import BDay

def bday_add(days: int):
    """Devuelve fecha objetivo sumando días hábiles a hoy."""
    return (Timestamp.today().normalize() + BDay(days)).date()