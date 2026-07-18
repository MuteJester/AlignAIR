import { Navigate, useParams } from "react-router-dom";
import { getLesson } from "../lessons/content";
import LessonPlayer from "../lessons/LessonPlayer";

export default function LessonPage() {
  const { track, slug } = useParams<{ track: string; slug: string }>();
  const lesson = track && slug ? getLesson(track, slug) : undefined;
  if (!lesson) return <Navigate to="/learn" replace />;
  return <LessonPlayer key={lesson.id} lesson={lesson} />;
}
