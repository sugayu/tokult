'''Wrapper utils of dataclasses.
'''

import dataclasses
from dataclasses import _FIELD, _FIELDS  # type: ignore


##
def fields(data: object) -> tuple[dataclasses.Field, ...]:
    '''Wrapper of "fields" function to avoid unreasonable mypy error.'''
    try:
        fields = getattr(data, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance') from None

    # Exclude pseudo-fields.  Note that fields is sorted by insertion
    # order, so the order of the tuple is as the fields were defined.
    return tuple(f for f in fields.values() if f._field_type is _FIELD)


def fieldnames(data: object) -> tuple[str, ...]:
    '''Return only field names.'''
    try:
        fields = getattr(data, _FIELDS)
    except AttributeError:
        raise TypeError('must be called with a dataclass type or instance') from None

    return tuple(f.name for f in fields.values() if f._field_type is _FIELD)
