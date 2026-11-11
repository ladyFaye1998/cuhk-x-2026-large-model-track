# Native encoder artifact

The original verification bundle uses a frozen native DSTformer initialization
at `assets/native/dstformer_best_epoch.bin` relative to the atomic-action
component. Its size is 29,571,972 bytes and SHA-256 is
`996f3b5fbb8e6fb00e7619bbad724995f0ab17692d0a3a0020cf941a5e551479`.
The recorded source is `github.com/cgy-lavine/CUHK-X-Code`, commit
`e5fb61aed31c1c850294037e4612be562881817d`, at
`SM/Code/skeleton/checkpoint/cuhkx_cross_trial/DSTformer_cross_trial_train.txt_resmapling_smallermodel_2nd_try/best_epoch.bin`.

Obtain this external initialization through an authorized source. The
teacher/student training code fits its downstream models from official observations
and generated pseudo-labels. The encoder remains frozen during student fitting.

See the [CUHK-X source and license notice](../../third_party/CUHK-X/NOTICE.md).
