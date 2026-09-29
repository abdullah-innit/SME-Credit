import "./globals.css";
import { Bricolage_Grotesque, Public_Sans } from "next/font/google";

const display = Bricolage_Grotesque({ subsets: ["latin"], variable: "--font-display" });
const body = Public_Sans({ subsets: ["latin"], variable: "--font-body" });

export const metadata = { title: "SME Credit Scrutiny", description: "Alternative-data credit assessment for SMEs" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable}`}>
      <body>{children}</body>
    </html>
  );
}
