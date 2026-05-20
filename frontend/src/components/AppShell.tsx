import { NavLink } from "react-router-dom";
import type { PropsWithChildren } from "react";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/analysis", label: "Analysis" },
  { to: "/cases", label: "Cases" },
  { to: "/method", label: "Method" },
];

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="min-h-screen bg-app-radial text-ink">
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-6 sm:px-6 lg:px-8">
        <header className="panel mb-6 overflow-hidden">
          <div className="flex flex-col gap-5 px-6 py-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <p className="panel-title mb-3">War News Bias Analyzer</p>
              <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
                Multimodal War News Bias Analyzer
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
                Enter one image and one passage to inspect a two-stage multimodal bias analysis pipeline:
                first generate observations, then produce cues and consistency.
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {navItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `nav-link ${isActive ? "nav-link-active" : ""}`
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        </header>
        <main className="flex-1">{children}</main>
      </div>
    </div>
  );
}
