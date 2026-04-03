# Permissions Guide — multiagent-framework-memory

After deploying the app, you must grant the app's **service principal (SP)** access to all resources the agent uses.

## Step 1: Get the SP Client ID

```bash
databricks apps get <your-app-name> --profile <profile> -o json | grep service_principal_client_id
```

Example output: `"service_principal_client_id": "a799151d-5efb-4fc9-a3dd-ebf8a07e609d"`

## Step 2: Unity Catalog Grants

Run these in a **Databricks SQL editor** or notebook:

```sql
-- Replace <sp-client-id> with your SP's client ID
-- Replace <catalog> and <schema> with your values

GRANT USE CATALOG ON CATALOG <catalog> TO `<sp-client-id>`;
GRANT USE SCHEMA ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
GRANT SELECT ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
GRANT EXECUTE ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
```

If your subagents use multiple schemas (e.g., one for tables, one for VS indexes), grant on each:

```sql
GRANT USE SCHEMA ON SCHEMA <catalog>.<second_schema> TO `<sp-client-id>`;
GRANT SELECT ON SCHEMA <catalog>.<second_schema> TO `<sp-client-id>`;
```

### Long-term Memory Table

If using long-term memory, the framework creates a Delta table `user_long_term_memory`. The SP needs MODIFY access:

```sql
-- Grant after the table is created (first request triggers creation)
GRANT MODIFY ON TABLE <catalog>.<schema>.user_long_term_memory TO `<sp-client-id>`;
GRANT SELECT ON TABLE <catalog>.<schema>.user_long_term_memory TO `<sp-client-id>`;
```

**Or** grant CREATE TABLE so the SP can create it automatically:

```sql
GRANT CREATE TABLE ON SCHEMA <catalog>.<schema> TO `<sp-client-id>`;
```

## Step 3: Workspace Resource Permissions

Grant via the **Databricks UI** (or REST API):

| Resource | Where | Permission |
|----------|-------|------------|
| **SQL Warehouse** | SQL Warehouses → your warehouse → Permissions | **Can Use** |
| **Genie Space** | Genie → your space → Share | **Can Run** |
| **Vector Search Endpoint** | Vector Search → Endpoints → your endpoint → Permissions | **Can Manage** |

### Via REST API (alternative)

```bash
SP="<sp-client-id>"
HOST="https://<your-workspace>"
TOKEN="<your-oauth-token>"

# Warehouse
curl -X PATCH "$HOST/api/2.0/permissions/sql/warehouses/<warehouse-id>" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"access_control_list\": [{\"service_principal_name\": \"$SP\", \"permission_level\": \"CAN_USE\"}]}"

# Genie Space
curl -X PATCH "$HOST/api/2.0/permissions/genie/<genie-space-id>" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"access_control_list\": [{\"service_principal_name\": \"$SP\", \"permission_level\": \"CAN_RUN\"}]}"

# Vector Search Endpoint (use endpoint UUID, not name)
curl -X PATCH "$HOST/api/2.0/permissions/vector-search-endpoints/<endpoint-uuid>" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"access_control_list\": [{\"service_principal_name\": \"$SP\", \"permission_level\": \"CAN_MANAGE\"}]}"
```

## Step 4: Lakebase Grants (Short-Term Memory)

Short-term memory uses `AsyncDatabricksSession` which stores conversation history in Lakebase Postgres tables (`agent_sessions`, `agent_messages`).

### Option A: Provisioned Lakebase (recommended)

Add the database resource in `databricks.yml`:
```yaml
resources:
  - name: "database"
    database:
      instance_name: "<your-lakebase-instance>"
      database_name: "postgres"
      permission: "CAN_CONNECT_AND_CREATE"
```

This automatically grants the SP credentials. **No manual SQL grants needed.**

### Option B: Autoscaling Lakebase

Autoscaling Lakebase requires manual setup:

1. **Add the postgres resource via API** (after each `bundle deploy`):
```bash
# Get existing resources
EXISTING=$(databricks apps get <app-name> -o json | jq -c '.resources // []')

# Append postgres
UPDATED=$(echo "$EXISTING" | jq -c '. + [{
  "name": "postgres",
  "postgres": {
    "branch": "projects/<project>/branches/<branch>",
    "database": "projects/<project>/branches/<branch>/databases/<db-id>",
    "permission": "CAN_CONNECT_AND_CREATE"
  }
}]')

# Patch
databricks api patch /api/2.0/apps/<app-name> --json "{\"resources\": $UPDATED}"
```

2. **Redeploy** to pick up credentials:
```bash
databricks apps deploy <app-name> --source-code-path <source-path>
```

3. **Grant Lakebase table permissions** (run in Lakebase SQL Editor):
```sql
-- Grant on existing tables
GRANT ALL ON ALL TABLES IN SCHEMA public TO "<sp-client-id>";
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO "<sp-client-id>";

-- Grant on future tables (so it survives re-deploys)
ALTER DEFAULT PRIVILEGES FOR ROLE "<your-email>" IN SCHEMA public
  GRANT ALL ON TABLES TO "<sp-client-id>";
ALTER DEFAULT PRIVILEGES FOR ROLE "<your-email>" IN SCHEMA public
  GRANT ALL ON SEQUENCES TO "<sp-client-id>";
```

Replace `<your-email>` with the workspace user who created the Lakebase project.

### Finding Lakebase IDs

```bash
# List projects
databricks api get /api/2.0/postgres/projects

# List branches
databricks api get /api/2.0/postgres/projects/<project>/branches

# List databases
databricks api get /api/2.0/postgres/projects/<project>/branches/<branch>/databases

# List endpoints (for connection hostname)
databricks api get /api/2.0/postgres/projects/<project>/branches/<branch>/endpoints
```

## Quick Reference: All Grants Summary

| Resource | Grant | Where | Required For |
|----------|-------|-------|-------------|
| UC Catalog | USE CATALOG | SQL | All subagents |
| UC Schema | USE SCHEMA, SELECT, EXECUTE | SQL | All subagents |
| SQL Warehouse | CAN_USE | UI/API | UC Functions, Long-term memory |
| Genie Space | CAN_RUN | UI/API | Genie subagent |
| VS Endpoint | CAN_MANAGE | UI/API | Vector Search subagent |
| Lakebase (provisioned) | CAN_CONNECT_AND_CREATE | databricks.yml | Short-term memory |
| Lakebase (autoscaling) | GRANT ALL ON TABLES | Lakebase SQL | Short-term memory |
| Memory Delta table | SELECT, MODIFY | SQL | Long-term memory |
