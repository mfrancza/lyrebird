# Repository settings for the agentic-development-workflow integration.
# https://github.com/mfrancza/agentic-development-workflow/blob/main/docs/adopting.md
#
# Usage (state is local and gitignored):
#   cd terraform
#   export GITHUB_TOKEN=$(gh auth token)   # needs `repo` scope, repo admin
#   terraform init && terraform plan && terraform apply

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.2"
    }
  }
}

provider "github" {
  owner = "mfrancza"
}

locals {
  developer_bot = "mfrancza-developer-agent[bot]"
  reviewer_bot  = "mfrancza-reviewer-agent[bot]"
}

# Adopt the existing repository and GitHub's default labels on first apply.
import {
  to = github_repository.this
  id = "lyrebird"
}

import {
  for_each = toset(["bug", "enhancement", "question"])
  to       = module.labels.github_issue_label.automation[each.key]
  id       = "lyrebird:${each.key}"
}

# Mirrors the current settings; only adopted so modules can reference it.
resource "github_repository" "this" {
  name        = "lyrebird"
  description = "DIY Neural Network Audio FX Modeling"
  visibility  = "public"

  has_issues   = true
  has_wiki     = true
  has_projects = true

  allow_merge_commit     = true
  allow_squash_merge     = true
  allow_rebase_merge     = true
  delete_branch_on_merge = false

  lifecycle {
    prevent_destroy = true
  }
}

module "labels" {
  source     = "git::https://github.com/mfrancza/agentic-development-workflow.git//terraform/modules/labels?ref=v0"
  repository = github_repository.this.name
}

module "agent_vars" {
  source          = "git::https://github.com/mfrancza/agentic-development-workflow.git//terraform/modules/agent-vars?ref=v0"
  repository      = github_repository.this.name
  agent_allowlist = ["mfrancza", "mfrancza-s-claude-code[bot]", local.developer_bot, local.reviewer_bot]
  default_model   = "sonnet"
  admin_assignees = ["mfrancza"]
  code_reviewers  = ["mfrancza"]
  auto_trigger_agents = {
    groom     = false
    design    = true
    developer = true
    review    = true
  }
}

module "branch_protection" {
  source     = "git::https://github.com/mfrancza/agentic-development-workflow.git//terraform/modules/branch-protection?ref=v0"
  repository = github_repository.this.name
}

module "security" {
  source     = "git::https://github.com/mfrancza/agentic-development-workflow.git//terraform/modules/security?ref=v0"
  repository = github_repository.this.name
}

module "actions_policy" {
  source     = "git::https://github.com/mfrancza/agentic-development-workflow.git//terraform/modules/actions-policy?ref=v0"
  repository = github_repository.this.name
  # Every non-GitHub-owned action must be listed here and SHA-pinned in the
  # workflow YAML. Update both together when bumping an action.
  patterns_allowed = [
    "mfrancza/agentic-development-workflow/.github/workflows/*@v0",
    "dorny/paths-filter@0e4a8c6effa4802afeda77dc8d303f8176d7dfad",
    "microsoft/setup-msbuild@6fb02220983dee41ce7ae257b6f4d8f9bf5ed4ce",
    "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65",
  ]
}
