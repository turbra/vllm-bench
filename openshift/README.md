# Deploy vLLM Bench Web UI to OpenShift

## Files

- `deployment.yaml` — Deploys the web app Pod(s)
- `service.yaml` — Exposes the app internally on port `8080`
- `route.yaml` — Exposes the app externally using **edge TLS termination** and **redirects HTTP → HTTPS**

## Prereqs

- You are logged into the target cluster:
  ```bash
  oc whoami
````

* You have a target namespace/project (create one if needed):

  ```bash
  oc new-project vllm-bench
  ```

## 1) Update the image in `deployment.yaml` (required)

Open `deployment.yaml` and update the container image reference:

```yaml
spec:
  template:
    spec:
      containers:
        - name: vllm-bench
          image: <REPLACE-ME>   # <-- update this
```

Replace `tagname` with the correct tag (or point to your internal registry path).

> This step is mandatory — if you don’t update the image, OpenShift will deploy whatever placeholder/tag is currently set.

## 2) Apply the manifests

From the directory containing the YAML files:

```bash
oc apply -f deployment.yaml
oc apply -f service.yaml
oc apply -f route.yaml
```

Or apply all at once:

```bash
oc apply -f deployment.yaml -f service.yaml -f route.yaml
```

## 3) Verify rollout

```bash
oc get deploy vllm-bench
oc rollout status deploy/vllm-bench
oc get pods -l app=vllm-bench
```

Check logs if needed:

```bash
oc logs deploy/vllm-bench
```

## 4) Get the Route URL

```bash
oc get route vllm-bench
```

Or extract just the host:

```bash
oc get route vllm-bench -o jsonpath='{.spec.host}{"\n"}'
```

Open in your browser:

* `https://<route-host>`

## Notes

### TLS / Redirect behavior

The Route is configured to:

* Use the cluster’s default TLS cert via **edge termination**
* Redirect insecure HTTP traffic to HTTPS:

  * `insecureEdgeTerminationPolicy: Redirect`

### Re-deploying after updates

If you push a new image tag:

1. Update the `image:` value in `deployment.yaml`
2. Re-apply:

   ```bash
   oc apply -f deployment.yaml
   oc rollout status deploy/vllm-bench
   ```

## Cleanup

```bash
oc delete -f route.yaml
oc delete -f service.yaml
oc delete -f deployment.yaml
```