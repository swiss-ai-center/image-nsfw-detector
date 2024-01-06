# Create a new service (generic) template

This repository contains the Python + FastAPI template to create a service
without a model or from an existing model compatible with the Core engine.

Please read the documentation at
<https://docs.swiss-ai-center.ch/how-to-guides/how-to-create-a-new-service> to
understand how to use this template.

## Guidelines

TODO: Add instructions on how to edit this template.

### Publishing and deploying using a CI/CD pipeline

This is the recommended way to publish and deploy your service if you have
access to GitHub Actions or GitLab CI.

TODO

### Publishing and deploying manually

This is the recommended way to publish and deploy your service if you do not
have access to GitHub Actions or GitLab CI or do not want to use these services.

TODO

## Checklist

These checklists allow you to ensure ensure everything is set up correctly.

### Common tasks

- [ ] Rename the project in the [`pyproject.toml`](./pyproject.toml) file
- [x] Add files that must be ignored to the [`.gitignore`](.gitignore) configuration file
- [ ] TODO

### Publishing and deploying using a CI/CD pipeline

> [!NOTE]  
> This checklist is specific to the _Publishing and deploying using a CI/CD
> pipeline_ section.

- [x] Add the environment variables
- [ ] TODO

### Publishing and deploying manually

> [!NOTE]  
> This checklist is specific to the _Publishing and deploying manually_ section.

- [x] Edit the [`.env`](.env) configuration file
- [ ] TODO
