"""SIH26174 - AI Human Activity Recognition for On-board BAS Experiments.

Package layout (see ``docs/DEVELOPMENT_PLAN.md`` for ownership):

* ``har.contracts``        frozen cross-track data contracts
* ``har.perception``       Track B - detection, pose, tracking, HOI
* ``har.protocol``         Track A - protocol model and sequence validator
* ``har.out``              Track C - event log, recorder, streamer, voice
* ``har.ui``               Track C - in-frame overlay and browser GUI
* ``har.app``              Track C - single CLI entrypoint
"""

from har.contracts import CONTRACT_VERSION

__version__ = CONTRACT_VERSION
