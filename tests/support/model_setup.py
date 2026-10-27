"""Shared reference-bound setup for tests that need the built-in protein class panel."""
from alignair.core import prepare_model
from alignair.reference import build_protein_reference


def protein_class_setup(dc_name='HUMAN_IGH_OGRDB', max_seq_length=192, *, has_d=True, **options):
    return prepare_model(build_protein_reference(dc_name), input_space='protein', label_space='protein_classes',
                         max_seq_length=max_seq_length, has_d=has_d, **options)
