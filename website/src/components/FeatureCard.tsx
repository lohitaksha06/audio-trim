interface FeatureCardProps {
  title: string;
  description: string;
  icon: string;
}

export default function FeatureCard({ title, description, icon }: FeatureCardProps) {
  return (
    <div className="group relative rounded-2xl border border-white/10 bg-white/5 p-5 sm:p-6 lg:p-7 transition-all duration-300 hover:border-neon-blue/30 hover:bg-white/10 hover:shadow-[0_0_30px_rgba(0,212,255,0.1)]">
      <div className="mb-4 text-3xl sm:text-4xl">{icon}</div>
      <h3 className="mb-2 text-lg sm:text-xl font-semibold text-white">{title}</h3>
      <p className="text-sm sm:text-base leading-relaxed text-white/60">{description}</p>
    </div>
  );
}
