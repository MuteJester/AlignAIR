import { Routes, Route, Navigate, useLocation } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import DocsPage from "./pages/DocsPage";
import { movedTo } from "./pages/docs/nav";

/** An address from before the restructure (the Learn lessons), sent to the page that now covers it. */
function Moved() {
  return <Navigate to={movedTo(useLocation().pathname) ?? "/docs"} replace />;
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/learn/*" element={<Moved />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:slug" element={<DocsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
