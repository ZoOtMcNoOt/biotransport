Configuration Classes
=====================

.. currentmodule:: biotransport

Multi-physics solver configurations are validated Python dataclasses that feed
the C++ numerical kernels.  Temperature-bearing bioheat names include ``_K``;
use ``BioheatCryotherapyConfig.from_celsius(...)`` for explicit Celsius input.
No default is a universal physiological recommendation.


Bioheat Cryotherapy
-------------------

.. autoclass:: BioheatCryotherapyConfig
   :members:


Tumor Drug Delivery
-------------------

.. autoclass:: TumorDrugDeliveryConfig
   :members:


Utilities
---------

.. autofunction:: get_parameter_ranges
