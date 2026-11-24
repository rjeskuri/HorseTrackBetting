## Ticket
Closes #<issue>

## Summary
<what changed and why, in two or three lines>

## Definition of done
- [ ] <copied from the ticket, checked when it holds>

## Tests
- `<test file>::<test>`: <what it proves>

Red run before implementation (<kind>, exit <code>):
```
<tail of harness-red check>
```

## Verify
```
<tail of the verify command, ending in VERIFY OK>
```

## Review
<findings by severity, and what was fixed>

## Experiment report (modeling PRs only)
Follow the [experiment report template](../docs/experiment-report.md); leave this section out if the PR changes no model.

### Hypothesis
<what you expected, and what result would show you wrong>

### Variants tried
<every variant, including failures, each with its CI>

### Alternatives rejected and why
<what was considered and dropped, with reasons>

### What this rules out
<what the result excludes>

### Follow-ups filed
<ticket numbers>

## Risks
<what could break and how it would show, or "none">
