import { signIn } from "@/lib/auth";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

export default function SignInPage() {
  return (
    <div className="flex min-h-[70vh] items-center justify-center">
      <Card className="w-full max-w-sm">
        <CardContent className="flex flex-col items-center gap-4 py-10 text-center">
          <h1 className="text-lg font-semibold">nar-web にサインイン</h1>
          <p className="text-sm text-muted-foreground">
            Google アカウントがあれば誰でも登録・ログインできます。
          </p>
          <form
            action={async () => {
              "use server";
              await signIn("google", { redirectTo: "/" });
            }}
          >
            <Button type="submit">Google でサインイン</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
