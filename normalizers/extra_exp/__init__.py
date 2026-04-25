"""Extra-experiment normalizer variants.

Modules in this package are used ONLY by experiments launched under
``scripts/extra_exp/`` via ``run_extra_exp.py`` / ``exp/exp_extra.py``.

They are not imported by the main runtime (``run_longExp.py`` /
``exp/exp_main.py``), so deleting this package is safe if you only want to
keep the main forecasting pipeline.
"""
