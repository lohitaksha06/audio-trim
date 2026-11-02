interface FeatureCardProps {
  title: string;
  description: string;
  icon: React.ReactNode;
  note?: string;
}

export default function FeatureCard({ title, description, icon, note }: FeatureCardProps) {
  return (
    <div className="border-t border-white/15 pt-5">
      <div className="mb-5 flex items-start justify-between gap-4">
        <div className="text-white/50">{icon}</div>
        {note && (
          <span className="font-mono text-[0.55rem] uppercase tracking-[0.2em] text-white/35">
            {note}
          </span>
        )}
      </div>
      <h3 className="font-display text-2xl font-bold leading-tight text-[#f2f1ec] sm:text-3xl">{title}</h3>
      <p className="mt-3 text-sm leading-relaxed text-white/55 sm:text-base">{description}</p>
    </div>
  );
}
