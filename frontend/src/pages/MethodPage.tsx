import { CUE_GROUPS, CUE_META, CUE_ORDER } from "../config/cues";

export function MethodPage() {
  return (
    <div className="space-y-6">
      <section className="grid gap-6 xl:grid-cols-2">
        <div className="panel p-6">
          <p className="panel-title mb-4">Method</p>
          <h2 className="text-2xl font-semibold text-slate-900">
            Two-stage structured analysis instead of a single classifier
          </h2>
          <div className="mt-6 space-y-4">
            <div className="rounded-3xl bg-slate-50 p-5">
              <div className="text-sm font-semibold text-slate-900">Stage 1</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                Given the image and article text, the system first generates
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  visual_observation
                </code>
                ,
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  textual_observation
                </code>
                and
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  joint_mechanism
                </code>
                .
              </p>
            </div>
            <div className="rounded-3xl bg-slate-50 p-5">
              <div className="text-sm font-semibold text-slate-900">Stage 2</div>
              <p className="mt-2 text-sm leading-6 text-slate-700">
                Based on the image, text, and observations, the system predicts 9 cues with
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  present
                </code>
                ,
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  score
                </code>
                , and
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  reason
                </code>
                , plus the image-text relationship
                <code className="mx-1 rounded bg-white px-1.5 py-0.5 text-xs">
                  consistency_dimension
                </code>
                .
              </p>
            </div>
          </div>
        </div>

        <div className="panel p-6">
          <p className="panel-title mb-4">How to Read the Output</p>
          <div className="space-y-4">
            {[
              "Start with consistency to judge whether the image and text reinforce, contrast, or supplement each other.",
              "Then inspect the active cues to locate whether the bias signal mainly comes from the image, the text, or their combination.",
              "Finally, read the observations to understand why the model formed the current attribution chain.",
            ].map((item) => (
              <div key={item} className="rounded-3xl bg-slate-50 p-5 text-sm leading-6 text-slate-700">
                {item}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel p-6">
        <p className="panel-title mb-4">Relationship Types</p>
        <div className="grid gap-4 xl:grid-cols-3">
          <div className="rounded-3xl bg-rose-50 p-5">
            <div className="text-sm font-semibold text-slate-900">Reinforcing</div>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              The image and text strengthen each other in the same evaluative direction.
            </p>
          </div>
          <div className="rounded-3xl bg-sky-50 p-5">
            <div className="text-sm font-semibold text-slate-900">Contrastive</div>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              The image and text pull in different directions, creating interpretive tension or perceptual conflict.
            </p>
          </div>
          <div className="rounded-3xl bg-slate-50 p-5">
            <div className="text-sm font-semibold text-slate-900">Supplementary</div>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              The image and text do not directly reinforce or conflict, but instead contribute different layers of information.
            </p>
          </div>
        </div>
      </section>

      <section className="panel p-6">
        <p className="panel-title mb-4">Cue Glossary</p>
        <div className="grid gap-4 xl:grid-cols-3">
          {CUE_GROUPS.map((group) => (
            <div key={group.key} className="rounded-3xl bg-slate-50 p-5">
              <div className="text-sm font-semibold text-slate-900">{group.label}</div>
              <p className="mt-2 text-sm leading-6 text-slate-600">
                {group.description}
              </p>
              <div className="mt-4 space-y-3">
                {CUE_ORDER.filter((cueCode) => CUE_META[cueCode].group === group.key).map(
                  (cueCode) => (
                    <div key={cueCode} className="rounded-2xl bg-white px-4 py-3">
                      <div className="text-sm font-medium text-slate-900">
                        {cueCode}
                      </div>
                      <div className="mt-1 text-sm text-slate-600">
                        {CUE_META[cueCode].label}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
