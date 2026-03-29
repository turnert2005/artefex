# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Artefex, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers or use GitHub's private vulnerability reporting
3. Include steps to reproduce and potential impact
4. Allow reasonable time for a fix before public disclosure

## Security Considerations

Artefex processes untrusted image files. The following precautions are in place:

- Image parsing is handled by Pillow, a well-maintained library with security patches
- Temporary files are cleaned up after processing
- The web UI uses FastAPI with standard security defaults
- No user data is stored or transmitted externally
- Neural models are loaded from local files only (no remote code execution)

## Dependencies

We monitor dependencies for known vulnerabilities. If you find a vulnerability in a dependency that affects Artefex, please report it.
