# Security policy

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability, leaked credential, or
private position data.

Email the maintainer at [cfw@pm.me](mailto:cfw@pm.me) with the subject
`[NautiPy security]`. Include:

- a description of the issue and its potential impact;
- the affected NautiPy version or commit;
- a minimal reproduction or proof of concept;
- relevant platform and dependency versions; and
- any mitigation or disclosure constraints you already know about.

Use synthetic coordinates and remove secrets whenever possible. If sensitive
material is necessary to reproduce the issue, describe it first and wait for a
safe transfer method.

The maintainer will acknowledge the report, investigate it privately, and
coordinate remediation and disclosure with the reporter. Please avoid public
discussion until a fix or agreed disclosure plan is available.

## Supported code

Security fixes target the current default branch and published package
versions covered by the [support policy](docs/SUPPORT.md). An unreleased
repository snapshot is development code, not a supported release.

Ordinary installation failures, parsing errors, or numerical discrepancies
that do not expose sensitive data or create a security impact belong in the
[public issue tracker](https://github.com/cafawo/NautiPy/issues).
